use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::{env, fs, path::PathBuf, process::Command};

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
    /// Rent a GPU, start a model on it and open a tunnel to it
    Up {
        /// One of the names from `vastx models`; prompts if left out
        model: Option<String>,
    },
    /// List the GPU instances vastx is running for you
    Ps,
    /// Reopen the tunnel to a running instance, restarting the model server if needed
    Connect {
        /// Instance id from `vastx ps`
        id: u64,
    },
    /// Destroy an instance so it stops billing
    Down {
        /// Instance id from `vastx ps`
        id: u64,
    },
}

// --- models ---

struct Model {
    name: &'static str,
    /// GGUF weights llama.cpp serves, pinned so an upstream change can't break a deployment.
    gguf_repo: &'static str,
    gguf_file: &'static str,
    gguf_revision: &'static str,
    /// The original repo, read for config.json: the architecture numbers sizing needs.
    base_repo: &'static str,
    base_revision: &'static str,
    license: &'static str,
}

const MODELS: [Model; 4] = [
    Model {
        name: "qwen2.5-1.5b",
        gguf_repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        gguf_revision: "91cad51170dc346986eccefdc2dd33a9da36ead9",
        base_repo: "Qwen/Qwen2.5-1.5B-Instruct",
        base_revision: "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        license: "apache-2.0",
    },
    Model {
        name: "qwen3-8b",
        gguf_repo: "Qwen/Qwen3-8B-GGUF",
        gguf_file: "Qwen3-8B-Q4_K_M.gguf",
        gguf_revision: "7c41481f57cb95916b40956ab2f0b139b296d974",
        base_repo: "Qwen/Qwen3-8B",
        base_revision: "b968826d9c46dd6066d109eabc6255188de91218",
        license: "apache-2.0",
    },
    Model {
        name: "mistral-7b",
        // Mistral publishes no GGUF; bartowski's conversion is the most used.
        gguf_repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        gguf_file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        gguf_revision: "61fd4167fff3ab01ee1cfe0da183fa27a944db48",
        base_repo: "mistralai/Mistral-7B-Instruct-v0.3",
        base_revision: "c170c708c41dac9275d15a8fff4eca08d52bab71",
        license: "apache-2.0",
    },
    Model {
        name: "deepseek-r1-7b",
        // DeepSeek publishes no GGUF; bartowski's conversion is the most used.
        gguf_repo: "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        gguf_file: "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        gguf_revision: "361004151d4f4f6b446dc5e6d46fbf4422a80d5f",
        base_repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        base_revision: "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        license: "mit",
    },
];

impl Model {
    fn gguf_url(&self) -> String {
        format!(
            "{HF}/{}/resolve/{}/{}",
            self.gguf_repo, self.gguf_revision, self.gguf_file
        )
    }
}

/// What vastx needs to know about a model before renting a GPU for it.
#[derive(Debug, PartialEq)]
struct ModelInfo {
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

impl ModelInfo {
    fn fetch(hf: &Http, m: &Model) -> Result<Self> {
        let tree = hf.get(&format!(
            "{HF}/api/models/{}/tree/{}",
            m.gguf_repo, m.gguf_revision
        ))?;
        let config = hf.get(&format!(
            "{HF}/{}/resolve/{}/config.json",
            m.base_repo, m.base_revision
        ))?;
        Self::from_parts(&tree, m.gguf_file, &config)
            .with_context(|| format!("reading {} from Hugging Face", m.name))
    }

    fn from_parts(tree: &Value, gguf_file: &str, config: &Value) -> Result<Self> {
        let weight_bytes = tree
            .as_array()
            .into_iter()
            .flatten()
            .find(|e| e["path"] == gguf_file)
            .and_then(|e| e["size"].as_u64())
            .with_context(|| format!("{gguf_file} is not in the repo"))?;
        Ok(Self {
            weight_bytes,
            arch: Arch::from_config(config)?,
        })
    }

    /// The context length vastx serves with: long enough for chat, short enough for small GPUs.
    fn context(&self) -> u64 {
        self.arch.max_context.min(8192)
    }

    /// GPU memory llama.cpp needs at `context()`: the GGUF weights, an fp16 KV cache for the
    /// full context, and 1 GiB for the CUDA context and compute buffers.
    fn vram_bytes(&self) -> u64 {
        let a = &self.arch;
        let kv = 2 * a.layers * a.kv_heads * a.head_dim * 2 * self.context();
        self.weight_bytes + kv + (1 << 30)
    }
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
            "{:<16} {:>6.1}GB {:>7.1}GB  {:<11} {}/{}",
            m.name,
            info.weight_bytes as f64 / GIB,
            info.vram_bytes() as f64 / GIB,
            m.license,
            m.gguf_repo,
            m.gguf_file
        );
    }
    Ok(())
}

// --- login ---

#[derive(Debug, PartialEq)]
struct Config {
    vast_api_key: String,
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
        Ok(v["vast_api_key"].as_str().map(|k| Self {
            vast_api_key: k.into(),
        }))
    }

    fn save(&self) -> Result<()> {
        use std::{io::Write, os::unix::fs::OpenOptionsExt};
        let path = Self::path()?;
        fs::create_dir_all(path.parent().context("config path has no parent")?)?;
        let body = json!({ "vast_api_key": self.vast_api_key });
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

    ensure_ssh_key(&vast)?;
    let config = Config {
        vast_api_key: vast.key,
    };
    config.save()?;
    eprintln!("Saved to {}.", Config::path()?.display());
    Ok(config)
}

/// vastx's own SSH key for rented machines. It has no passphrase because vastx runs commands
/// over SSH without a terminal to prompt in, and the user's own key may be locked.
fn ssh_key_path() -> Result<PathBuf> {
    Ok(home()?.join(".ssh").join("vastx_ed25519"))
}

/// Creates vastx's SSH key if it's missing and makes sure vast.ai has it, so new instances
/// accept it. Returns the private key's path.
fn ensure_ssh_key(vast: &Vast) -> Result<PathBuf> {
    let p = ssh_key_path()?;
    if !p.exists() {
        fs::create_dir_all(p.parent().context("no .ssh directory")?)?;
        let status = Command::new("ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-C", "vastx", "-f"])
            .arg(&p)
            .status()
            .context("running ssh-keygen")?;
        if !status.success() {
            bail!("ssh-keygen failed ({status})");
        }
        eprintln!("Created SSH key {} for rented machines.", p.display());
    }
    let public = fs::read_to_string(p.with_extension("pub"))?;
    if !registered(&vast.get("/ssh/")?, &public) {
        vast.post("/ssh/", &json!({ "ssh_key": public.trim() }))?;
        eprintln!("Added SSH key {} to your vast.ai account.", p.display());
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
        self.send(url, || self.auth(self.agent.get(url)).call())
    }

    fn post(&self, url: &str, body: &Value) -> Result<Value> {
        self.send(url, || self.auth(self.agent.post(url)).send_json(body))
    }

    fn put(&self, url: &str, body: &Value) -> Result<Value> {
        self.send(url, || self.auth(self.agent.put(url)).send_json(body))
    }

    fn delete(&self, url: &str) -> Result<Value> {
        self.send(url, || self.auth(self.agent.delete(url)).call())
    }

    fn auth<B>(&self, r: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.token {
            Some(t) => r.header("Authorization", format!("Bearer {t}")),
            None => r,
        }
    }

    /// Sends a request, waiting and retrying when rate limited. vast.ai allows only a few
    /// requests per second and says how long to wait in `retry_after`.
    fn send(
        &self,
        url: &str,
        request: impl Fn() -> Result<ureq::http::Response<ureq::Body>, ureq::Error>,
    ) -> Result<Value> {
        for attempt in 1.. {
            let mut res = request()?;
            let status = res.status();
            let text = res.body_mut().read_to_string()?;
            if status == 429 && attempt < 6 {
                let wait = serde_json::from_str::<Value>(&text)
                    .ok()
                    .and_then(|v| v["retry_after"].as_f64())
                    .unwrap_or(5.0)
                    .clamp(1.0, 60.0);
                std::thread::sleep(std::time::Duration::from_secs_f64(wait));
                continue;
            }
            if !status.is_success() {
                let short: String = text.chars().take(300).collect();
                bail!("{url}: {status}: {short}");
            }
            return serde_json::from_str(&text)
                .with_context(|| format!("{url}: response is not JSON"));
        }
        unreachable!()
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
    /// A bare image starts in about 30 s; vastx installs llama.cpp itself (see `setup_script`).
    pub const IMAGE: &str = "ubuntu:24.04";
    const BILLING_PAGE: &str = "https://console.vast.ai/billing/";
    /// Marks instances vastx created, so `ps` and `down` never touch anything else.
    const LABEL_PREFIX: &str = "vastx:";

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

        pub fn put(&self, path: &str, body: &Value) -> Result<Value> {
            self.http.put(&format!("{API}{path}"), body)
        }

        /// Rents `offer` and returns the new instance's id.
        pub fn rent(&self, offer: &Offer, model: &str, disk_gb: u64) -> Result<u64> {
            let res = self
                .put(&format!("/asks/{}/", offer.id), &rent_body(model, disk_gb))
                .map_err(|e| {
                    if e.to_string().contains("insufficient_credit") {
                        anyhow::anyhow!(
                            "your vast.ai account has no credit; add some at {BILLING_PAGE}"
                        )
                    } else {
                        e
                    }
                })?;
            res["new_contract"]
                .as_u64()
                .with_context(|| format!("vast.ai didn't return an instance id: {res}"))
        }

        /// The instances vastx created on this account.
        pub fn instances(&self) -> Result<Vec<Instance>> {
            let res = self.get("/instances/?owner=me")?;
            Ok(res["instances"]
                .as_array()
                .context("vast.ai returned no instances list")?
                .iter()
                .filter_map(Instance::parse)
                .collect())
        }

        pub fn instance(&self, id: u64) -> Result<Option<Instance>> {
            Ok(self.instances()?.into_iter().find(|i| i.id == id))
        }

        pub fn destroy(&self, id: u64) -> Result<()> {
            self.http.delete(&format!("{API}/instances/{id}/"))?;
            Ok(())
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
            // Hosts in China usually can't reach Docker Hub or Hugging Face.
            "geolocation": {"notin": ["CN"]},
            // Ampere or newer: fast fp16 and a GPU recent enough for any engine vastx adds.
            "compute_cap": {"gte": 800},
            // Driver new enough for llama.cpp's CUDA 12.8 build.
            "cuda_max_good": {"gte": 12.8},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": 50,
            // Prices include storage, so ask for what will actually be allocated.
            "allocated_storage": need.disk_gb,
        })
    }

    pub(super) fn rent_body(model: &str, disk_gb: u64) -> Value {
        json!({
            "client_id": "me",
            "image": IMAGE,
            "disk": disk_gb,
            "label": format!("{LABEL_PREFIX}{model}"),
            // SSH straight to the host when it has open ports, through vast's proxy otherwise.
            "runtype": "ssh_direc ssh_proxy",
            // Fail right away if someone else rented the offer first, instead of queueing.
            "cancel_unavail": true,
        })
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Instance {
        pub id: u64,
        pub model: String,
        pub gpu_name: String,
        pub dph: f64,
        /// vast's state: "loading" while the image pulls, then "running".
        pub status: String,
        /// vast's latest progress note, like "Pulling image".
        pub status_msg: String,
        pub ssh: Option<(String, u16)>,
    }

    impl Instance {
        /// None for instances vastx didn't create.
        pub(super) fn parse(v: &Value) -> Option<Self> {
            let model = v["label"].as_str()?.strip_prefix(LABEL_PREFIX)?;
            let direct = v["ports"]["22/tcp"][0]["HostPort"]
                .as_str()
                .and_then(|p| p.parse().ok())
                .zip(v["public_ipaddr"].as_str().map(|h| h.trim().to_string()));
            let proxy = v["ssh_port"]
                .as_u64()
                .and_then(|p| u16::try_from(p).ok())
                .zip(v["ssh_host"].as_str().map(String::from));
            Some(Self {
                id: v["id"].as_u64()?,
                model: model.into(),
                gpu_name: v["gpu_name"].as_str().unwrap_or("?").into(),
                dph: v["dph_total"].as_f64().unwrap_or(0.0),
                status: v["actual_status"].as_str().unwrap_or("starting").into(),
                status_msg: v["status_msg"].as_str().unwrap_or("").trim().into(),
                ssh: direct.or(proxy).map(|(port, host)| (host, port)),
            })
        }
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

use vast::{Instance, Need, Offer, Vast};

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
        .map(|m| format!("{:<16} {}", m.name, m.base_repo))
        .collect();
    let i = dialoguer::Select::new()
        .with_prompt("Model")
        .items(&items)
        .default(0)
        .interact()?;
    Ok(&MODELS[i])
}

/// Disk to allocate: the weights plus room for the OS and llama.cpp (about 1 GB).
fn disk_gb(info: &ModelInfo) -> u64 {
    info.weight_bytes.div_ceil(1_000_000_000) + 10
}

fn pick_offer(vast: &Vast, model: &Model) -> Result<(Offer, Need)> {
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
    Ok((shown[i].clone(), need))
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
    let (offer, _) = pick_offer(&Vast::new(config.vast_api_key), model)?;
    eprintln!(
        "Picked {} in {} at ${:.3}/hr (offer {}). Rent it with `vastx up {}`.",
        offer.gpu_name, offer.location, offer.dph, offer.id, model.name
    );
    Ok(())
}

// --- renting ---

fn cmd_up(model: Option<&str>) -> Result<()> {
    let config = require_login()?;
    let vast = Vast::new(config.vast_api_key.clone());
    let model = pick_model(model)?;
    let (offer, need) = pick_offer(&vast, model)?;
    let key = ensure_ssh_key(&vast)?;
    let id = vast.rent(&offer, model.name, need.disk_gb)?;
    eprintln!(
        "Rented instance {id}: {} in {} at ${:.3}/hr.",
        offer.gpu_name, offer.location, offer.dph
    );
    eprintln!("It bills until you run `vastx down {id}`, even if you stop vastx now.");
    let started = std::time::Instant::now();
    let ready = wait_for_ssh(&vast, id, &key)
        .and_then(|(host, port)| setup(model, &key, &host, port).map(|()| (host, port)));
    match ready {
        Ok((host, port)) => {
            eprintln!(
                "{} is up after {:.0}s.",
                model.name,
                started.elapsed().as_secs_f64()
            );
            tunnel(model, id, &key, &host, port)
        }
        Err(e) => {
            eprintln!("Instance {id} didn't come up: {e:#}");
            let destroy = dialoguer::Confirm::new()
                .with_prompt(format!("Destroy instance {id} so it stops billing?"))
                .default(true)
                .interact()?;
            if destroy {
                vast.destroy(id)?;
                eprintln!("Destroyed instance {id}.");
            } else {
                eprintln!("Left it running; `vastx down {id}` destroys it.");
            }
            Err(e)
        }
    }
}

/// Polls until the instance runs and accepts SSH. The first start pulls the ~10 GB vLLM image,
/// which can take several minutes on a slow host.
fn wait_for_ssh(vast: &Vast, id: u64, key: &std::path::Path) -> Result<(String, u16)> {
    let now = std::time::Instant::now;
    let deadline = now() + std::time::Duration::from_secs(20 * 60);
    // Status and SSH notes are tracked apart so they don't reprint each other every poll.
    let (mut last_status, mut last_ssh) = (String::new(), String::new());
    let mut running_since = None;
    loop {
        let inst = vast
            .instance(id)?
            .with_context(|| format!("instance {id} disappeared from your account"))?;
        let note = status_note(&inst);
        if note != last_status {
            eprintln!("  {note}");
            last_status = note;
        }
        if inst.status == "exited" {
            bail!("the instance exited ({})", inst.status_msg);
        }
        // Docker errors (like a host that can't reach the registry) don't fix themselves.
        if inst.status_msg.contains("Error response from daemon") {
            bail!("the host couldn't start the image: {}", inst.status_msg);
        }
        if let (true, Some((host, port))) = (inst.status == "running", inst.ssh.clone()) {
            let since = *running_since.get_or_insert_with(now);
            let out = Command::new("ssh")
                .args(ssh_command(key, &host, port).into_iter().skip(1))
                .args(["-o", "ConnectTimeout=10", "true"])
                .stdin(std::process::Stdio::null())
                .output()
                .context("running ssh")?;
            if out.status.success() {
                return Ok((host, port));
            }
            // vast prints a welcome banner on stderr; the last line is ssh's actual error.
            let err = String::from_utf8_lossy(&out.stderr);
            let reason = err.lines().last().unwrap_or("no response");
            let note = format!("waiting for SSH: {reason}");
            if note != last_ssh {
                eprintln!("  {note}");
                last_ssh = note;
            }
            // Healthy hosts accept the key within seconds of running. Some never do.
            if reason.contains("Permission denied")
                && since.elapsed() > std::time::Duration::from_secs(3 * 60)
            {
                bail!("the host never accepted the SSH key");
            }
        }
        if now() > deadline {
            bail!("timed out after 20 minutes");
        }
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}

/// One line of progress. Docker reports each image layer separately, so those collapse
/// into "pulling image".
fn status_note(i: &Instance) -> String {
    let msg = &i.status_msg;
    let layer = msg.len() >= 13
        && msg.as_bytes()[12] == b':'
        && msg[..12].bytes().all(|b| b.is_ascii_hexdigit());
    match (msg.is_empty(), layer) {
        (true, _) => i.status.clone(),
        (_, true) => format!("{}: pulling image", i.status),
        _ => format!("{}: {msg}", i.status),
    }
}

/// vast reuses host:port pairs across instances, so a stored host key would soon mismatch.
/// These machines are short-lived and reached by IP from vast's API, so host keys are not kept.
fn ssh_command(key: &std::path::Path, host: &str, port: u16) -> Vec<String> {
    [
        "ssh",
        "-i",
        &key.to_string_lossy(),
        "-p",
        &port.to_string(),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "BatchMode=yes",
        // Only vastx's key: agent keys tried first can use up the server's auth attempts.
        "-o",
        "IdentitiesOnly=yes",
        &format!("root@{host}"),
    ]
    .map(String::from)
    .to_vec()
}

// --- serving ---

/// Pinned llama.cpp release; its CUDA 12.8 build runs on any driver vast's filter allows.
const LLAMA_RELEASE: &str = "b12777";
/// llama-server's port on the instance. It listens on localhost only and is reached through
/// the SSH tunnel, so nothing is exposed to the internet.
const REMOTE_PORT: u16 = 8080;

/// Shell script that installs llama.cpp, downloads the model and starts llama-server. Every step
/// is skipped when already done, so `vastx connect` can rerun it safely.
fn setup_script(m: &Model, context: u64) -> String {
    let rel = format!("https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}");
    let build = "bin-ubuntu-cuda-12.8-x64.tar.gz";
    let gguf = format!("/workspace/models/{}", m.gguf_file);
    format!(
        r#"set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v curl >/dev/null || ! dpkg -s libgomp1 >/dev/null 2>&1; then
  echo "Installing system packages"
  apt-get update -qq && apt-get install -y -qq curl ca-certificates libgomp1 >/dev/null
fi
if [ ! -x /opt/llama/llama-server ]; then
  echo "Installing llama.cpp {LLAMA_RELEASE}"
  mkdir -p /opt/llama
  curl -fsSL --retry 3 {rel}/llama-{LLAMA_RELEASE}-{build} | tar xz -C /opt/llama --strip-components=1
  curl -fsSL --retry 3 {rel}/cudart-llama-{LLAMA_RELEASE}-{build} | tar xz -C /opt/llama --strip-components=1
fi
if [ ! -f {gguf} ]; then
  echo "Downloading {file} from Hugging Face"
  mkdir -p /workspace/models
  curl -fL --retry 3 -C - -# -o {gguf}.part {url}
  mv {gguf}.part {gguf}
fi
if ! curl -sf http://127.0.0.1:{port}/health >/dev/null; then
  echo "Starting llama-server"
  pkill -f /opt/llama/llama-server || true
  cd /opt/llama
  LD_LIBRARY_PATH=/opt/llama nohup ./llama-server -m {gguf} --alias {name} -ngl 999 \
    -c {context} --host 127.0.0.1 --port {port} >/workspace/llama-server.log 2>&1 &
  for _ in $(seq 1 600); do
    curl -sf http://127.0.0.1:{port}/health >/dev/null && exit 0
    kill -0 $! 2>/dev/null || break
    sleep 0.5
  done
  echo "llama-server didn't start; last lines of /workspace/llama-server.log:" >&2
  tail -20 /workspace/llama-server.log >&2
  exit 1
fi
"#,
        file = m.gguf_file,
        url = m.gguf_url(),
        name = m.name,
        port = REMOTE_PORT,
    )
}

/// Runs `setup_script` on the instance, showing its progress in this terminal.
fn setup(m: &Model, key: &std::path::Path, host: &str, port: u16) -> Result<()> {
    use std::io::Write;
    let info = ModelInfo::fetch(&Http::new(env::var("HF_TOKEN").ok(), true), m)?;
    let mut child = Command::new("ssh")
        .args(ssh_command(key, host, port).into_iter().skip(1))
        .arg("bash -s")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("running ssh")?;
    child
        .stdin
        .take()
        .context("ssh has no stdin")?
        .write_all(setup_script(m, info.context()).as_bytes())?;
    let status = child.wait()?;
    if !status.success() {
        bail!("setting up {} failed ({status})", m.name);
    }
    Ok(())
}

/// Forwards llama-server to a free local port and keeps the tunnel open until Ctrl-C.
fn tunnel(m: &Model, id: u64, key: &std::path::Path, host: &str, port: u16) -> Result<()> {
    let local = free_port(REMOTE_PORT)?;
    let mut child = Command::new("ssh")
        .args(ssh_command(key, host, port).into_iter().skip(1))
        .args([
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
        ])
        .args(["-L", &format!("{local}:127.0.0.1:{REMOTE_PORT}")])
        .spawn()
        .context("running ssh")?;
    eprintln!(
        "\nServing {name} at http://localhost:{local}/v1 (OpenAI-compatible). Try:\n",
        name = m.name
    );
    println!(
        "curl -s http://localhost:{local}/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'",
        m.name
    );
    eprintln!(
        "\nCtrl-C closes the tunnel. The GPU keeps billing until `vastx down {id}`; \
         `vastx connect {id}` reopens the tunnel."
    );
    let status = child.wait()?;
    if !status.success() {
        bail!("the tunnel closed ({status})");
    }
    Ok(())
}

/// `preferred` if nothing is listening there, otherwise the next free port above it.
fn free_port(preferred: u16) -> Result<u16> {
    (preferred..preferred.saturating_add(100))
        .find(|p| std::net::TcpListener::bind(("127.0.0.1", *p)).is_ok())
        .context("no free local port for the tunnel")
}

fn cmd_connect(id: u64) -> Result<()> {
    let config = require_login()?;
    let vast = Vast::new(config.vast_api_key);
    let inst = vast
        .instance(id)?
        .with_context(|| format!("no vastx instance {id}; `vastx ps` lists them"))?;
    let model = MODELS
        .iter()
        .find(|m| m.name == inst.model)
        .with_context(|| {
            format!(
                "instance {id} runs {}, which vastx doesn't know",
                inst.model
            )
        })?;
    let (host, port) = inst
        .ssh
        .with_context(|| format!("instance {id} isn't reachable yet ({})", inst.status))?;
    let key = ensure_ssh_key(&vast)?;
    setup(model, &key, &host, port)?;
    tunnel(model, id, &key, &host, port)
}

fn cmd_ps() -> Result<()> {
    let config = require_login()?;
    let instances = Vast::new(config.vast_api_key).instances()?;
    if instances.is_empty() {
        eprintln!("No vastx instances running.");
        return Ok(());
    }
    println!(
        "{:<10} {:<16} {:<16} {:<9} {:>9}  SSH",
        "ID", "MODEL", "GPU", "STATUS", "$/HR"
    );
    for i in &instances {
        println!("{}", ps_row(i));
    }
    Ok(())
}

fn ps_row(i: &Instance) -> String {
    let ssh = match &i.ssh {
        Some((host, port)) => format!("root@{host} -p {port}"),
        None => "-".into(),
    };
    format!(
        "{:<10} {:<16} {:<16} {:<9} {:>9.3}  {ssh}",
        i.id, i.model, i.gpu_name, i.status, i.dph
    )
}

fn cmd_down(id: u64) -> Result<()> {
    let config = require_login()?;
    let vast = Vast::new(config.vast_api_key);
    // Only instances vastx created; anything else on the account is left alone.
    if vast.instance(id)?.is_none() {
        bail!("no vastx instance {id}; `vastx ps` lists them");
    }
    vast.destroy(id)?;
    eprintln!("Destroyed instance {id}.");
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Some(Cmd::Login) => drop(cmd_login()?),
        Some(Cmd::Models) => cmd_models()?,
        Some(Cmd::Gpus { model }) => cmd_gpus(model.as_deref())?,
        Some(Cmd::Up { model }) => cmd_up(model.as_deref())?,
        Some(Cmd::Ps) => cmd_ps()?,
        Some(Cmd::Connect { id }) => cmd_connect(id)?,
        Some(Cmd::Down { id }) => cmd_down(id)?,
        None => cmd_up(None)?,
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
    fn gguf_size_comes_from_the_tree() {
        let tree = json!([
            {"type": "file", "path": "Qwen3-8B-Q4_K_M.gguf", "size": 5_027_783_488u64},
            {"type": "file", "path": "Qwen3-8B-Q8_0.gguf", "size": 8_709_518_624u64}
        ]);
        let info = ModelInfo::from_parts(&tree, "Qwen3-8B-Q4_K_M.gguf", &qwen_config()).unwrap();
        assert_eq!(info.weight_bytes, 5_027_783_488);
        assert!(ModelInfo::from_parts(&tree, "missing.gguf", &qwen_config()).is_err());
    }

    #[test]
    fn vram_estimate() {
        let info = ModelInfo {
            weight_bytes: 5 * (1 << 30),
            arch: Arch::from_config(&qwen_config()).unwrap(),
        };
        assert_eq!(info.context(), 8192);
        // KV: 2 * 28 layers * 4 heads * 128 dim * 2 bytes * 8192 tokens = 448 MiB.
        assert_eq!(info.vram_bytes(), 6 * (1 << 30) + 448 * (1 << 20));
    }

    #[test]
    fn setup_script_is_pinned_and_idempotent() {
        let m = &MODELS[1];
        let s = setup_script(m, 8192);
        assert!(
            s.contains("releases/download/b12777/llama-b12777-bin-ubuntu-cuda-12.8-x64.tar.gz")
        );
        assert!(s.contains(
            "releases/download/b12777/cudart-llama-b12777-bin-ubuntu-cuda-12.8-x64.tar.gz"
        ));
        assert!(s.contains(&format!(
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/{}/Qwen3-8B-Q4_K_M.gguf",
            m.gguf_revision
        )));
        assert!(s.contains("--alias qwen3-8b"));
        assert!(s.contains("-c 8192 --host 127.0.0.1 --port 8080"));
        for guard in [
            "if [ ! -x /opt/llama/llama-server ]",
            "if [ ! -f /workspace/models/",
            "if ! curl -sf",
        ] {
            assert!(s.contains(guard), "{guard}");
        }
    }

    #[test]
    fn free_port_skips_busy_ones() {
        let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let p = busy.local_addr().unwrap().port();
        assert_ne!(free_port(p).unwrap(), p);
    }

    #[test]
    fn model_names_are_unique() {
        let mut names: Vec<_> = MODELS.iter().map(|m| m.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), MODELS.len());
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
        assert_eq!(q["geolocation"]["notin"][0], "CN");
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
    fn rent_body_labels_and_pins_image() {
        let b = vast::rent_body("qwen3-8b-awq", 47);
        assert_eq!(b["label"], "vastx:qwen3-8b-awq");
        assert_eq!(b["image"], vast::IMAGE);
        assert_eq!(b["disk"], 47);
        assert_eq!(b["cancel_unavail"], true);
    }

    #[test]
    fn parses_instances() {
        let direct = json!({"id": 7, "label": "vastx:mistral-7b", "gpu_name": "RTX 3090",
            "dph_total": 0.24, "actual_status": "running", "status_msg": "success, running\n",
            "public_ipaddr": "1.2.3.4\n", "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40022"}]},
            "ssh_host": "ssh4.vast.ai", "ssh_port": 12345});
        let i = Instance::parse(&direct).unwrap();
        assert_eq!(i.model, "mistral-7b");
        assert_eq!(i.status_msg, "success, running");
        assert_eq!(i.ssh, Some(("1.2.3.4".into(), 40022)));
        assert_eq!(
            ps_row(&i),
            "7          mistral-7b       RTX 3090         running       0.240  root@1.2.3.4 -p 40022"
        );

        let loading = json!({"id": 8, "label": "vastx:qwen2.5-1.5b", "actual_status": null,
            "ssh_host": "ssh4.vast.ai", "ssh_port": 12345});
        let i = Instance::parse(&loading).unwrap();
        assert_eq!(i.status, "starting");
        assert_eq!(i.ssh, Some(("ssh4.vast.ai".into(), 12345)));

        for other in [
            json!({"id": 9, "label": "my training run"}),
            json!({"id": 10, "label": null}),
        ] {
            assert!(Instance::parse(&other).is_none());
        }
    }

    #[test]
    fn status_note_collapses_layers() {
        let mut i = Instance::parse(
            &json!({"id": 1, "label": "vastx:m", "actual_status": "loading",
            "status_msg": "da997ea44e13: Verifying Checksum"}),
        )
        .unwrap();
        assert_eq!(status_note(&i), "loading: pulling image");
        i.status_msg = "f8ca4e3fe64a:".into();
        assert_eq!(status_note(&i), "loading: pulling image");
        i.status_msg = "success, running vllm/vllm-openai_v0.33.0/ssh".into();
        assert_eq!(
            status_note(&i),
            "loading: success, running vllm/vllm-openai_v0.33.0/ssh"
        );
        i.status_msg.clear();
        assert_eq!(status_note(&i), "loading");
    }

    #[test]
    fn disk_covers_weights_and_llama() {
        let info = ModelInfo {
            weight_bytes: 5_027_783_488,
            arch: Arch::from_config(&qwen_config()).unwrap(),
        };
        assert_eq!(disk_gb(&info), 16);
    }

    #[test]
    fn ssh_key_registration_check() {
        let keys = json!([{"id": 1, "public_key": "ssh-ed25519 AAAAC3Nza1 laptop"}]);
        assert!(registered(&keys, "ssh-ed25519 AAAAC3Nza1 vastx\n"));
        assert!(!registered(&keys, "ssh-ed25519 AAAAOTHER vastx"));
        assert!(!registered(&keys, "garbage"));
    }
}
