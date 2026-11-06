use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};
use std::{env, io, path::PathBuf, process::Command};

const S5CMD_VERSION: &str = "2.3.0";
const CF_API: &str = "https://api.cloudflare.com/client/v4";
const EXCLUDES: [&str; 3] = ["*/.git/*", "*/.cache/*", "*.DS_Store"];

#[derive(Parser)]
#[command(
    version,
    about = "Tooling for renting cloud GPUs for LLM training and inference"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Mirror a local model directory (or single file) to the bucket
    Push {
        path: PathBuf,
        /// Defaults to the path's base name
        name: Option<String>,
        #[command(flatten)]
        tuning: Tuning,
    },
    /// Download a model using your own credentials
    Pull {
        name: String,
        /// Defaults to ./<name>
        dest: Option<String>,
        #[command(flatten)]
        tuning: Tuning,
    },
    /// Print a one-line download command for a GPU instance, using read-only credentials scoped to the model
    Share {
        name: String,
        /// Credential lifetime, at most 168
        #[arg(long, default_value_t = 24)]
        hours: u64,
        /// Defaults to /workspace/models/<name>
        #[arg(long)]
        dest: Option<String>,
        #[arg(long, value_enum, default_value_t = Arch::Amd64)]
        arch: Arch,
        #[command(flatten)]
        tuning: Tuning,
    },
    /// List models and their sizes
    Ls,
    /// Delete a model
    Rm {
        name: String,
        #[arg(short, long)]
        yes: bool,
    },
}

#[derive(clap::Args)]
struct Tuning {
    /// Files transferred in parallel [push: 4, pull: 32]
    #[arg(short, long)]
    workers: Option<u32>,
    /// Parts transferred in parallel per file [push: 8, pull: 16]
    #[arg(short, long)]
    concurrency: Option<u32>,
    /// Part size in MiB
    #[arg(short, long, default_value_t = 64)]
    part_mib: u32,
}

impl Tuning {
    fn argv(&self, r2: &R2, op: &str, (workers, concurrency): (u32, u32)) -> Vec<String> {
        let mut v = r2.argv(Some(self.workers.unwrap_or(workers)), op);
        let c = self.concurrency.unwrap_or(concurrency);
        v.extend([
            "-c".into(),
            c.to_string(),
            "-p".into(),
            self.part_mib.to_string(),
        ]);
        v
    }
}

const PUSH: (u32, u32) = (4, 8);
const PULL: (u32, u32) = (32, 16);

#[derive(Clone, Copy, ValueEnum)]
enum Arch {
    Amd64,
    Arm64,
}

struct R2 {
    account: String,
    key: String,
    secret: String,
    bucket: String,
    endpoint: String,
}

impl R2 {
    fn from_env() -> Result<Self> {
        let var = |k: &str| env::var(k).unwrap_or_default();
        let account = var("VASTX_R2_ACCOUNT_ID");
        let endpoint = match var("VASTX_S3_ENDPOINT") {
            e if e.is_empty() => format!("https://{account}.r2.cloudflarestorage.com"),
            e => e,
        };
        let r2 = R2 {
            key: var("VASTX_R2_ACCESS_KEY_ID"),
            secret: var("VASTX_R2_SECRET_ACCESS_KEY"),
            bucket: var("VASTX_R2_BUCKET"),
            account,
            endpoint,
        };
        if r2.key.is_empty()
            || r2.secret.is_empty()
            || r2.bucket.is_empty()
            || r2.endpoint.contains("https://.")
        {
            bail!(
                "set VASTX_R2_ACCOUNT_ID, VASTX_R2_ACCESS_KEY_ID, VASTX_R2_SECRET_ACCESS_KEY and VASTX_R2_BUCKET"
            );
        }
        Ok(r2)
    }

    fn argv(&self, workers: Option<u32>, op: &str) -> Vec<String> {
        let mut v = vec!["--endpoint-url".into(), self.endpoint.clone()];
        if let Some(w) = workers {
            v.extend(["--numworkers".into(), w.to_string()]);
        }
        v.push(op.into());
        v
    }

    fn url(&self, name: &str) -> String {
        format!("s3://{}/{name}/", self.bucket)
    }

    fn s5(&self, args: &[String]) -> Command {
        let mut cmd = Command::new("s5cmd");
        cmd.args(args)
            .env("AWS_ACCESS_KEY_ID", &self.key)
            .env("AWS_SECRET_ACCESS_KEY", &self.secret)
            .env("AWS_REGION", "auto");
        cmd
    }

    fn run(&self, args: &[String]) -> Result<()> {
        let status = self.s5(args).status().map_err(s5cmd_missing)?;
        if !status.success() {
            bail!("s5cmd failed ({status})");
        }
        Ok(())
    }

    fn output(&self, args: &[String]) -> Result<String> {
        let out = self.s5(args).output().map_err(s5cmd_missing)?;
        if !out.status.success() {
            bail!(
                "s5cmd failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn temp_credentials(&self, name: &str, hours: u64) -> Result<[String; 3]> {
        let token = env::var("CLOUDFLARE_API_TOKEN").context(
            "share needs CLOUDFLARE_API_TOKEN, a Cloudflare API token with R2 write access",
        )?;
        let api = env::var("VASTX_CF_API").unwrap_or_else(|_| CF_API.into());
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build()
            .into();
        let v: Value = agent
            .post(format!(
                "{api}/accounts/{}/r2/temp-access-credentials",
                self.account
            ))
            .header("Authorization", format!("Bearer {token}"))
            .send_json(json!({
                "bucket": self.bucket,
                "parentAccessKeyId": self.key,
                "permission": "object-read-only",
                "ttlSeconds": hours * 3600,
                "prefixes": [format!("{name}/")],
            }))?
            .body_mut()
            .read_json()?;
        parse_credentials(&v)
    }
}

fn parse_credentials(v: &Value) -> Result<[String; 3]> {
    if v["success"] != true {
        bail!("Cloudflare API error: {}", v["errors"]);
    }
    let r = &v["result"];
    let creds = ["accessKeyId", "secretAccessKey", "sessionToken"]
        .map(|k| r[k].as_str().unwrap_or_default().to_owned());
    if creds.iter().any(String::is_empty) {
        bail!("Cloudflare API returned incomplete credentials: {r}");
    }
    Ok(creds)
}

fn s5cmd_missing(e: io::Error) -> anyhow::Error {
    match e.kind() {
        io::ErrorKind::NotFound => anyhow::anyhow!(
            "s5cmd not found; install it with `brew install s5cmd` or from github.com/peak/s5cmd"
        ),
        _ => e.into(),
    }
}

fn check_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    if !ok {
        bail!("invalid model name {name:?}: use letters, digits, '.', '_' and '-'");
    }
    Ok(())
}

fn quote(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./:=".contains(c))
    {
        return s.into();
    }
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn bootstrap(creds: &[String; 3], argv: &[String], arch: Arch) -> String {
    let asset = match arch {
        Arch::Amd64 => "64bit",
        Arch::Arm64 => "arm64",
    };
    let v = S5CMD_VERSION;
    let install = format!(
        "command -v s5cmd >/dev/null || curl -fsSL https://github.com/peak/s5cmd/releases/download/v{v}/s5cmd_{v}_Linux-{asset}.tar.gz | tar -xz -C /usr/local/bin s5cmd"
    );
    let env = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]
    .iter()
    .zip(creds)
    .map(|(k, v)| format!("{k}={}", quote(v)))
    .collect::<Vec<_>>()
    .join(" ");
    let args: Vec<_> = argv.iter().map(|a| quote(a)).collect();
    format!(
        "{{ {install}; }} && {env} AWS_REGION=auto s5cmd {}",
        args.join(" ")
    )
}

fn confirm(prompt: &str) -> Result<bool> {
    eprint!("{prompt} [y/N] ");
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let r2 = R2::from_env()?;
    match cli.cmd {
        Cmd::Push { path, name, tuning } => {
            let name = match name {
                Some(n) => n,
                None => path
                    .canonicalize()?
                    .file_name()
                    .context("can't infer a name; pass one")?
                    .to_string_lossy()
                    .into(),
            };
            check_name(&name)?;
            let src = path.to_string_lossy().into_owned();
            let mut argv;
            if path.is_dir() {
                argv = tuning.argv(&r2, "sync", PUSH);
                argv.push("--delete".into());
                for e in EXCLUDES {
                    argv.extend(["--exclude".into(), e.into()]);
                }
                argv.push(format!("{}/", src.trim_end_matches('/')));
            } else {
                argv = tuning.argv(&r2, "cp", PUSH);
                argv.push(src);
            }
            argv.push(r2.url(&name));
            r2.run(&argv)?;
            eprintln!("pushed {name}; next: vastx share {name}");
        }
        Cmd::Pull { name, dest, tuning } => {
            check_name(&name)?;
            let mut argv = tuning.argv(&r2, "sync", PULL);
            argv.extend([
                r2.url(&name) + "*",
                dest.unwrap_or_else(|| name.clone()) + "/",
            ]);
            r2.run(&argv)?;
        }
        Cmd::Share {
            name,
            hours,
            dest,
            arch,
            tuning,
        } => {
            check_name(&name)?;
            if !(1..=168).contains(&hours) {
                bail!("--hours must be between 1 and 168");
            }
            let creds = r2.temp_credentials(&name, hours)?;
            let mut argv = tuning.argv(&r2, "sync", PULL);
            argv.extend([
                r2.url(&name) + "*",
                dest.unwrap_or_else(|| format!("/workspace/models/{name}")) + "/",
            ]);
            println!("{}", bootstrap(&creds, &argv, arch));
            eprintln!("credentials are read-only, limited to {name}/, and expire in {hours}h");
        }
        Cmd::Ls => {
            let mut ls = r2.argv(None, "ls");
            ls.push(format!("s3://{}/", r2.bucket));
            for line in r2.output(&ls)?.lines() {
                let Some(name) = line
                    .trim()
                    .strip_prefix("DIR")
                    .map(|s| s.trim().trim_end_matches('/'))
                else {
                    continue;
                };
                let mut du = r2.argv(None, "du");
                du.extend(["-H".into(), r2.url(name) + "*"]);
                let du = r2.output(&du)?;
                println!(
                    "{name:<40} {}",
                    du.split(':').next().unwrap_or_default().trim()
                );
            }
        }
        Cmd::Rm { name, yes } => {
            check_name(&name)?;
            if !yes && !confirm(&format!("delete {} from {}?", name, r2.bucket))? {
                bail!("aborted");
            }
            let mut rm = r2.argv(None, "rm");
            rm.push(r2.url(&name) + "*");
            r2.run(&rm)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        for ok in ["llama-3.1-8b", "run_1.ckpt"] {
            assert!(check_name(ok).is_ok(), "{ok}");
        }
        for bad in ["", ".hidden", "a/b", "..", "a b", "a*"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn quoting() {
        assert_eq!(quote("s3://b/m/"), "s3://b/m/");
        assert_eq!(quote("s3://b/m/*"), "'s3://b/m/*'");
        assert_eq!(quote("a'b"), r"'a'\''b'");
    }

    #[test]
    fn credentials() {
        let ok = json!({"success": true, "result": {"accessKeyId": "a", "secretAccessKey": "s", "sessionToken": "t"}});
        assert_eq!(parse_credentials(&ok).unwrap(), ["a", "s", "t"]);
        let err = json!({"success": false, "errors": [{"code": 10000, "message": "Authentication error"}]});
        assert!(
            parse_credentials(&err)
                .unwrap_err()
                .to_string()
                .contains("Authentication error")
        );
    }
}
