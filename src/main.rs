use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::{collections::HashSet, env, fs, path::PathBuf, process::Command};

const VAST_KEYS_PAGE: &str = "https://console.vast.ai/manage-keys/";
const HF: &str = "https://huggingface.co";
const GIB: f64 = (1u64 << 30) as f64;

#[derive(Parser)]
#[command(
    version,
    about = "Run open-weight models on rented GPUs. Run with no command to get started."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Connect your vast.ai account and register an SSH key with it
    Login,
    /// List the available models with their size and the GPU memory they need
    Models,
    /// Find GPU offers that fit a model and pick one
    Gpus {
        /// One of the names from `vastx models`; prompts if left out
        model: Option<String>,
    },
}

// --- models ---

struct Model {
    name: &'static str,
    repo: &'static str,
    /// Pinned so an upstream change can't break a deployment.
    revision: &'static str,
    license: &'static str,
}

const MODELS: [Model; 4] = [
    Model {
        name: "qwen2.5-1.5b",
        repo: "Qwen/Qwen2.5-1.5B-Instruct",
        revision: "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        license: "apache-2.0",
    },
    Model {
        name: "qwen3-8b-awq",
        repo: "Qwen/Qwen3-8B-AWQ",
        revision: "4da05a8edb55c6046cce958586c33b61da07bb79",
        license: "apache-2.0",
    },
    Model {
        name: "mistral-7b",
        repo: "mistralai/Mistral-7B-Instruct-v0.3",
        revision: "c170c708c41dac9275d15a8fff4eca08d52bab71",
        license: "apache-2.0",
    },
    Model {
        name: "deepseek-r1-7b",
        repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        revision: "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        license: "mit",
    },
];

/// What vastx needs to know about a model before renting a GPU for it.
#[derive(Debug, PartialEq)]
struct ModelInfo {
    /// Files to download: the weights vLLM loads plus config and tokenizer files.
    files: Vec<(String, u64)>,
    weight_bytes: u64,
    arch: Arch,
}

#[derive(Debug, PartialEq)]
struct Arch {
    layers: u64,
    kv_heads: u64,
    head_dim: u64,
    max_context: u64,
}

const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

impl ModelInfo {
    fn fetch(hf: &Http, m: &Model) -> Result<Self> {
        let base = format!("{HF}/{}/resolve/{}", m.repo, m.revision);
        let tree = hf.get(&format!(
            "{HF}/api/models/{}/tree/{}?recursive=true",
            m.repo, m.revision
        ))?;
        let index = if tree_files(&tree).any(|(p, _)| p == SAFETENSORS_INDEX) {
            Some(hf.get(&format!("{base}/{SAFETENSORS_INDEX}"))?)
        } else {
            None
        };
        let config = hf.get(&format!("{base}/config.json"))?;
        Self::from_parts(&tree, index.as_ref(), &config)
            .with_context(|| format!("reading {} from Hugging Face", m.repo))
    }

    fn from_parts(tree: &Value, index: Option<&Value>, config: &Value) -> Result<Self> {
        // Some repos ship the weights twice (Mistral has consolidated.safetensors next to the
        // HF shards). When there's an index, only the shards it lists are weights.
        let weights: HashSet<&str> = match index {
            Some(ix) => ix["weight_map"]
                .as_object()
                .context("safetensors index has no weight_map")?
                .values()
                .filter_map(Value::as_str)
                .collect(),
            None => tree_files(tree)
                .map(|(p, _)| p)
                .filter(|p| *p == "model.safetensors")
                .collect(),
        };
        if weights.is_empty() {
            bail!("no safetensors weights found");
        }
        let support = |p: &str| {
            !p.contains('/')
                && !p.starts_with("consolidated")
                && [".json", ".model", ".tiktoken", ".txt"]
                    .iter()
                    .any(|ext| p.ends_with(ext))
        };
        let files: Vec<(String, u64)> = tree_files(tree)
            .filter(|(p, _)| weights.contains(p) || support(p))
            .map(|(p, s)| (p.to_string(), s))
            .collect();
        let weight_bytes = files
            .iter()
            .filter(|(p, _)| weights.contains(p.as_str()))
            .map(|(_, s)| s)
            .sum();
        Ok(Self {
            files,
            weight_bytes,
            arch: Arch::from_config(config)?,
        })
    }

    /// The context length vastx serves with: long enough for chat, short enough for small GPUs.
    fn context(&self) -> u64 {
        self.arch.max_context.min(8192)
    }

    /// GPU memory needed to serve with vLLM at `context()`: weights, an fp16 KV cache for one
    /// full sequence, and 2 GiB for activations and CUDA graphs, at vLLM's default 90% memory
    /// utilization. Uses file bytes, so quantized weights need no special case.
    fn vram_bytes(&self) -> u64 {
        let a = &self.arch;
        let kv = 2 * a.layers * a.kv_heads * a.head_dim * 2 * self.context();
        ((self.weight_bytes + kv + 2 * (1 << 30)) as f64 / 0.9) as u64
    }
}

fn tree_files(tree: &Value) -> impl Iterator<Item = (&str, u64)> {
    tree.as_array()
        .into_iter()
        .flatten()
        .filter(|e| e["type"] == "file")
        .filter_map(|e| Some((e["path"].as_str()?, e["size"].as_u64()?)))
}

impl Arch {
    fn from_config(config: &Value) -> Result<Self> {
        // Multimodal configs nest the language model's numbers under text_config.
        let c = if config.get("num_hidden_layers").is_some() {
            config
        } else {
            &config["text_config"]
        };
        let num = |k: &str| c[k].as_u64();
        let need = |k: &str| num(k).with_context(|| format!("config.json has no {k}"));
        let heads = need("num_attention_heads")?;
        Ok(Self {
            layers: need("num_hidden_layers")?,
            kv_heads: num("num_key_value_heads").unwrap_or(heads),
            head_dim: match num("head_dim") {
                Some(d) => d,
                None => need("hidden_size")? / heads,
            },
            max_context: need("max_position_embeddings")?,
        })
    }
}

fn cmd_models() -> Result<()> {
    let hf = Http::new(env::var("HF_TOKEN").ok(), true);
    println!(
        "{:<16} {:>8} {:>9}  {:<11} SOURCE",
        "MODEL", "SIZE", "GPU MEM", "LICENSE"
    );
    for m in &MODELS {
        let info = ModelInfo::fetch(&hf, m)?;
        println!(
            "{:<16} {:>6.1}GB {:>7.1}GB  {:<11} {}",
            m.name,
            info.weight_bytes as f64 / GIB,
            info.vram_bytes() as f64 / GIB,
            m.license,
            m.repo
        );
    }
    Ok(())
}

// --- login ---

#[derive(Debug, PartialEq)]
struct Config {
    vast_api_key: String,
    /// Private key used to reach rented machines; its .pub is registered with vast.
    ssh_key: PathBuf,
}

impl Config {
    fn path() -> Result<PathBuf> {
        let base = match env::var_os("XDG_CONFIG_HOME") {
            Some(d) if !d.is_empty() => PathBuf::from(d),
            _ => home()?.join(".config"),
        };
        Ok(base.join("vastx").join("config.json"))
    }

    fn load() -> Result<Option<Self>> {
        let path = Self::path()?;
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let v: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(match (v["vast_api_key"].as_str(), v["ssh_key"].as_str()) {
            (Some(k), Some(s)) => Some(Self {
                vast_api_key: k.into(),
                ssh_key: s.into(),
            }),
            _ => None,
        })
    }

    fn save(&self) -> Result<()> {
        use std::{io::Write, os::unix::fs::OpenOptionsExt};
        let path = Self::path()?;
        fs::create_dir_all(path.parent().context("config path has no parent")?)?;
        let body = json!({ "vast_api_key": self.vast_api_key, "ssh_key": self.ssh_key });
        // The file holds an API key, so only the owner can read it.
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
            .with_context(|| format!("writing {}", path.display()))?;
        writeln!(f, "{}", serde_json::to_string_pretty(&body)?)?;
        Ok(())
    }
}

fn home() -> Result<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")
}

fn cmd_login() -> Result<Config> {
    eprintln!("vastx rents GPUs through your vast.ai account.");
    eprintln!("Create an API key at {VAST_KEYS_PAGE} (opening it now).");
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = Command::new(opener).arg(VAST_KEYS_PAGE).status();

    let vast = loop {
        let key = dialoguer::Password::new()
            .with_prompt("Paste your vast.ai API key")
            .interact()?;
        let vast = Vast::new(key.trim().to_string());
        match vast.get("/users/current/") {
            Ok(user) => {
                let who = user["email"].as_str().or(user["username"].as_str());
                eprintln!(
                    "Logged in{}.",
                    who.map(|w| format!(" as {w}")).unwrap_or_default()
                );
                break vast;
            }
            Err(e) => eprintln!("That key didn't work: {e:#}"),
        }
    };

    let ssh_key = ssh_key()?;
    let public = fs::read_to_string(ssh_key.with_extension("pub"))?;
    if registered(&vast.get("/ssh/")?, &public) {
        eprintln!(
            "SSH key {} is already on your vast.ai account.",
            ssh_key.display()
        );
    } else {
        vast.post("/ssh/", &json!({ "ssh_key": public.trim() }))?;
        eprintln!(
            "Added SSH key {} to your vast.ai account.",
            ssh_key.display()
        );
    }

    let config = Config {
        vast_api_key: vast.key,
        ssh_key,
    };
    config.save()?;
    eprintln!("Saved to {}.", Config::path()?.display());
    Ok(config)
}

/// The user's ed25519 or RSA key if they have one, otherwise a new key just for vastx.
fn ssh_key() -> Result<PathBuf> {
    let dir = home()?.join(".ssh");
    for name in ["id_ed25519", "id_rsa"] {
        let p = dir.join(name);
        if p.exists() && p.with_extension("pub").exists() {
            return Ok(p);
        }
    }
    let p = dir.join("vastx_ed25519");
    if !p.exists() {
        fs::create_dir_all(&dir)?;
        let status = Command::new("ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-C", "vastx", "-f"])
            .arg(&p)
            .status()
            .context("running ssh-keygen")?;
        if !status.success() {
            bail!("ssh-keygen failed ({status})");
        }
    }
    Ok(p)
}

/// Whether vast's key list already holds this public key. Compares only the base64 body,
/// since vast may store the key without its comment.
fn registered(keys: &Value, public: &str) -> bool {
    match public.split_whitespace().nth(1) {
        Some(body) => keys.to_string().contains(body),
        None => false,
    }
}

// --- http ---

struct Http {
    agent: ureq::Agent,
    token: Option<String>,
}

impl Http {
    /// `follow_redirects` is off for vast.ai: ureq drops the Authorization header on redirect,
    /// so a redirect would surface as a misleading "requires login" error.
    fn new(token: Option<String>, follow_redirects: bool) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(if follow_redirects { 10 } else { 0 })
            .build()
            .into();
        Self { agent, token }
    }

    fn get(&self, url: &str) -> Result<Value> {
        let mut r = self.agent.get(url);
        if let Some(t) = &self.token {
            r = r.header("Authorization", format!("Bearer {t}"));
        }
        Self::read(url, r.call()?)
    }

    fn post(&self, url: &str, body: &Value) -> Result<Value> {
        let mut r = self.agent.post(url);
        if let Some(t) = &self.token {
            r = r.header("Authorization", format!("Bearer {t}"));
        }
        Self::read(url, r.send_json(body)?)
    }

    fn read(url: &str, mut res: ureq::http::Response<ureq::Body>) -> Result<Value> {
        let status = res.status();
        let text = res.body_mut().read_to_string()?;
        if !status.is_success() {
            let short: String = text.chars().take(300).collect();
            bail!("{url}: {status}: {short}");
        }
        serde_json::from_str(&text).with_context(|| format!("{url}: response is not JSON"))
    }
}

// --- vast.ai ---

mod vast {
    //! vast.ai API client: account calls and GPU offer search.

    use crate::Http;
    use anyhow::{Context, Result};
    use serde_json::{Value, json};
    use std::fmt;

    const API: &str = "https://console.vast.ai/api/v0";

    pub struct Vast {
        pub key: String,
        http: Http,
    }

    impl Vast {
        pub fn new(key: String) -> Self {
            Self {
                // No redirects: ureq drops the Authorization header on redirect, which would
                // surface as a misleading "requires login". Paths need their trailing slash.
                http: Http::new(Some(key.clone()), false),
                key,
            }
        }

        pub fn get(&self, path: &str) -> Result<Value> {
            self.http.get(&format!("{API}{path}"))
        }

        pub fn post(&self, path: &str, body: &Value) -> Result<Value> {
            self.http.post(&format!("{API}{path}"), body)
        }

        /// Rentable single-GPU offers that meet `need`, cheapest first.
        pub fn offers(&self, need: &Need) -> Result<Vec<Offer>> {
            let res = self.post("/bundles/", &offer_query(need))?;
            let mut offers: Vec<Offer> = res["offers"]
                .as_array()
                .context("vast.ai offer search returned no offers list")?
                .iter()
                .filter_map(Offer::parse)
                .collect();
            rank(&mut offers);
            Ok(offers)
        }
    }

    /// What a machine must have to serve a model.
    pub struct Need {
        pub gpu_ram_mib: u64,
        pub disk_gb: u64,
    }

    pub(super) fn offer_query(need: &Need) -> Value {
        json!({
            "verified": {"eq": true},
            "external": {"eq": false},
            "rentable": {"eq": true},
            "rented": {"eq": false},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": need.gpu_ram_mib},
            "disk_space": {"gte": need.disk_gb},
            // Weights come over the network, so a slow link wastes paid GPU time.
            "inet_down": {"gte": 500},
            "reliability": {"gte": 0.98},
            // Ampere or newer: current vLLM needs it for bf16 and its AWQ kernels.
            "compute_cap": {"gte": 800},
            // Driver new enough for the CUDA 12.8 builds of the vLLM image.
            "cuda_max_good": {"gte": 12.8},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": 50,
            // Prices include storage, so ask for what will actually be allocated.
            "allocated_storage": need.disk_gb,
        })
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Offer {
        pub id: u64,
        pub gpu_name: String,
        pub gpu_ram_mib: u64,
        /// Total price per hour, including the requested storage.
        pub dph: f64,
        pub inet_down_mbps: f64,
        pub location: String,
    }

    impl Offer {
        pub(super) fn parse(v: &Value) -> Option<Self> {
            Some(Self {
                id: v["id"].as_u64()?,
                gpu_name: v["gpu_name"].as_str()?.into(),
                gpu_ram_mib: v["gpu_ram"].as_f64()? as u64,
                dph: v["dph_total"].as_f64()?,
                inet_down_mbps: v["inet_down"].as_f64().unwrap_or(0.0),
                location: v["geolocation"].as_str().unwrap_or("unknown").into(),
            })
        }
    }

    impl fmt::Display for Offer {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "{:<16} {:>3}GB  ${:.3}/hr  {:>5.0} Mbps  {}",
                self.gpu_name,
                self.gpu_ram_mib.div_ceil(1024),
                self.dph,
                self.inet_down_mbps,
                self.location
            )
        }
    }

    /// Cheapest first; at the same price, the faster download wins.
    pub(super) fn rank(offers: &mut [Offer]) {
        offers.sort_by(|a, b| {
            a.dph
                .total_cmp(&b.dph)
                .then(b.inet_down_mbps.total_cmp(&a.inet_down_mbps))
        });
    }
}

use vast::{Need, Offer, Vast};

// --- choosing a model and a GPU ---

fn pick_model(name: Option<&str>) -> Result<&'static Model> {
    if let Some(name) = name {
        return MODELS.iter().find(|m| m.name == name).with_context(|| {
            let names: Vec<_> = MODELS.iter().map(|m| m.name).collect();
            format!("unknown model {name:?}; choose one of {}", names.join(", "))
        });
    }
    let items: Vec<String> = MODELS
        .iter()
        .map(|m| format!("{:<16} {}", m.name, m.repo))
        .collect();
    let i = dialoguer::Select::new()
        .with_prompt("Model")
        .items(&items)
        .default(0)
        .interact()?;
    Ok(&MODELS[i])
}

/// Disk to allocate: the weights, the vLLM image (about 20 GB unpacked) and HF's download cache.
fn disk_gb(info: &ModelInfo) -> u64 {
    info.weight_bytes.div_ceil(1_000_000_000) + 40
}

fn pick_offer(vast: &Vast, model: &Model) -> Result<Offer> {
    let info = ModelInfo::fetch(&Http::new(env::var("HF_TOKEN").ok(), true), model)?;
    let need = Need {
        gpu_ram_mib: info.vram_bytes().div_ceil(1 << 20),
        disk_gb: disk_gb(&info),
    };
    eprintln!(
        "{} needs {:.1} GB of GPU memory and {} GB of disk.",
        model.name,
        info.vram_bytes() as f64 / GIB,
        need.disk_gb
    );
    let offers = vast.offers(&need)?;
    if offers.is_empty() {
        bail!(
            "no vast.ai offers fit {} right now; try again in a few minutes",
            model.name
        );
    }
    let shown = &offers[..offers.len().min(5)];
    let i = dialoguer::Select::new()
        .with_prompt("GPU (cheapest first)")
        .items(shown)
        .default(0)
        .interact()?;
    Ok(shown[i].clone())
}

fn require_login() -> Result<Config> {
    match Config::load()? {
        Some(c) => Ok(c),
        None => cmd_login(),
    }
}

fn cmd_gpus(model: Option<&str>) -> Result<()> {
    let config = require_login()?;
    let model = pick_model(model)?;
    let offer = pick_offer(&Vast::new(config.vast_api_key), model)?;
    eprintln!(
        "Picked {} in {} at ${:.3}/hr (offer {}). Renting it comes in the next version.",
        offer.gpu_name, offer.location, offer.dph, offer.id
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Some(Cmd::Login) => drop(cmd_login()?),
        Some(Cmd::Models) => cmd_models()?,
        Some(Cmd::Gpus { model }) => cmd_gpus(model.as_deref())?,
        None => cmd_gpus(None)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_config() -> Value {
        // Trimmed from Qwen/Qwen2.5-7B-Instruct's config.json.
        json!({
            "hidden_size": 3584, "num_attention_heads": 28, "num_key_value_heads": 4,
            "num_hidden_layers": 28, "max_position_embeddings": 32768
        })
    }

    #[test]
    fn arch_falls_back_and_unwraps_text_config() {
        let a = Arch::from_config(&qwen_config()).unwrap();
        assert_eq!(
            a,
            Arch {
                layers: 28,
                kv_heads: 4,
                head_dim: 128,
                max_context: 32768
            }
        );
        let nested = json!({ "architectures": ["X"], "text_config": {
            "head_dim": 256, "num_attention_heads": 8, "num_hidden_layers": 4,
            "max_position_embeddings": 4096
        }});
        let a = Arch::from_config(&nested).unwrap();
        assert_eq!((a.kv_heads, a.head_dim, a.max_context), (8, 256, 4096));
        assert!(Arch::from_config(&json!({"num_hidden_layers": 2})).is_err());
    }

    #[test]
    fn index_decides_weights_and_skips_duplicates() {
        // Shape of mistralai/Mistral-7B-Instruct-v0.3: consolidated weights next to HF shards.
        let tree = json!([
            {"type": "file", "path": "consolidated.safetensors", "size": 14_500},
            {"type": "file", "path": "model-00001-of-00002.safetensors", "size": 5_000},
            {"type": "file", "path": "model-00002-of-00002.safetensors", "size": 4_500},
            {"type": "file", "path": SAFETENSORS_INDEX, "size": 20},
            {"type": "file", "path": "config.json", "size": 1},
            {"type": "file", "path": "tokenizer.model", "size": 2},
            {"type": "file", "path": "README.md", "size": 4},
            {"type": "file", "path": "original/params.json", "size": 5},
            {"type": "directory", "path": "original", "size": 0}
        ]);
        let index = json!({"weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00001-of-00002.safetensors",
            "c": "model-00002-of-00002.safetensors"
        }});
        let info = ModelInfo::from_parts(&tree, Some(&index), &qwen_config()).unwrap();
        assert_eq!(info.weight_bytes, 9_500);
        let names: Vec<_> = info.files.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            names,
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                SAFETENSORS_INDEX,
                "config.json",
                "tokenizer.model",
            ]
        );
    }

    #[test]
    fn single_file_weights() {
        let tree = json!([
            {"type": "file", "path": "model.safetensors", "size": 3_000},
            {"type": "file", "path": "config.json", "size": 1}
        ]);
        let info = ModelInfo::from_parts(&tree, None, &qwen_config()).unwrap();
        assert_eq!(info.weight_bytes, 3_000);
        let empty = json!([{"type": "file", "path": "config.json", "size": 1}]);
        assert!(ModelInfo::from_parts(&empty, None, &qwen_config()).is_err());
    }

    #[test]
    fn vram_estimate() {
        let info = ModelInfo {
            files: vec![],
            weight_bytes: 15 * (1 << 30),
            arch: Arch::from_config(&qwen_config()).unwrap(),
        };
        assert_eq!(info.context(), 8192);
        // KV: 2 * 28 layers * 4 heads * 128 dim * 2 bytes * 8192 tokens = 448 MiB.
        let expected = (15.0 * GIB + 448.0 * 1024.0 * 1024.0 + 2.0 * GIB) / 0.9;
        assert_eq!(info.vram_bytes(), expected as u64);
    }

    #[test]
    fn offer_query_carries_requirements() {
        let q = vast::offer_query(&Need {
            gpu_ram_mib: 10_000,
            disk_gb: 46,
        });
        assert_eq!(q["gpu_ram"]["gte"], 10_000);
        assert_eq!(q["disk_space"]["gte"], 46);
        assert_eq!(q["allocated_storage"], 46);
        assert_eq!(q["num_gpus"]["eq"], 1);
        assert_eq!(q["compute_cap"]["gte"], 800);
    }

    #[test]
    fn parses_and_ranks_offers() {
        // Trimmed from a live /bundles/ response.
        let offers = json!([
            {"id": 61537938, "gpu_name": "RTX 3060", "gpu_ram": 12288, "dph_total": 0.078,
             "inet_down": 1434.8, "geolocation": "Poland, PL"},
            {"id": 60586610, "gpu_name": "RTX 5060 Ti", "gpu_ram": 16311, "dph_total": 0.071,
             "inet_down": 926.1, "geolocation": "Malaysia, MY"},
            {"id": 58397639, "gpu_name": "RTX 3060", "gpu_ram": 12288, "dph_total": 0.071,
             "inet_down": 867.8},
            {"id": "broken"}
        ]);
        let mut parsed: Vec<Offer> = offers
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Offer::parse)
            .collect();
        vast::rank(&mut parsed);
        let ids: Vec<u64> = parsed.iter().map(|o| o.id).collect();
        assert_eq!(ids, [60586610, 58397639, 61537938]);
        assert_eq!(parsed[1].location, "unknown");
        assert_eq!(
            parsed[0].to_string(),
            "RTX 5060 Ti       16GB  $0.071/hr    926 Mbps  Malaysia, MY"
        );
    }

    #[test]
    fn disk_covers_weights_and_image() {
        let info = ModelInfo {
            files: vec![],
            weight_bytes: 15_200_000_000,
            arch: Arch::from_config(&qwen_config()).unwrap(),
        };
        assert_eq!(disk_gb(&info), 56);
    }

    #[test]
    fn ssh_key_registration_check() {
        let keys = json!([{"id": 1, "public_key": "ssh-ed25519 AAAAC3Nza1 laptop"}]);
        assert!(registered(&keys, "ssh-ed25519 AAAAC3Nza1 vastx\n"));
        assert!(!registered(&keys, "ssh-ed25519 AAAAOTHER vastx"));
        assert!(!registered(&keys, "garbage"));
    }
}
