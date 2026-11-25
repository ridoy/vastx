#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.32"]
# ///
"""Publish models to the vastx catalog bucket (maintainer only).

    publish.py add hf:Qwen/Qwen2.5-7B-Instruct [--name N] [--revision SHA]
    publish.py add ./local-dir --name N [--license L]
    publish.py ls
    publish.py rm NAME

Needs s5cmd and VASTX_R2_ACCOUNT_ID, VASTX_R2_ACCESS_KEY_ID, VASTX_R2_SECRET_ACCESS_KEY.
VASTX_R2_BUCKET defaults to "models".
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Only what vLLM needs to serve; skips duplicate .bin/.pth/GGUF weights.
HF_PATTERNS = ["*.safetensors", "*.json", "*.model", "*.tiktoken", "*.txt", "LICENSE*"]
SKIP_DIRS = {".git", ".cache"}
MANIFEST = "vastx.json"
CATALOG = "catalog.json"


def check_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_\-][A-Za-z0-9._\-]*", name):
        sys.exit(f"invalid model name {name!r}: use letters, digits, '.', '_' and '-'")


# --- reading a model directory ---


def safetensors_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def weights_info(files):
    """Total parameter count and the dtype holding the most parameters."""
    params, by_dtype = 0, Counter()
    for f in files:
        for key, t in safetensors_header(f).items():
            if key == "__metadata__":
                continue
            count = 1
            for d in t["shape"]:
                count *= d
            params += count
            by_dtype[t["dtype"].lower()] += count
    if not params:
        sys.exit("no safetensors weights found")
    return params, by_dtype.most_common(1)[0][0]


def arch_info(config):
    """The numbers vastx needs to size GPUs and launch vLLM."""
    c = config.get("text_config", config) if "num_hidden_layers" not in config else config
    heads = c["num_attention_heads"]
    return {
        "layers": c["num_hidden_layers"],
        "heads": heads,
        "kv_heads": c.get("num_key_value_heads") or heads,
        "head_dim": c.get("head_dim") or c["hidden_size"] // heads,
        "max_context": c["max_position_embeddings"],
    }


def readme_license(root):
    readme = root / "README.md"
    if not readme.exists():
        return None
    m = re.match(r"---\n(.*?)\n---", readme.read_text(errors="replace"), re.S)
    if m and (lic := re.search(r"^license:\s*['\"]?([^'\"\n]+)", m.group(1), re.M)):
        return lic.group(1).strip()
    return None


def model_files(root):
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name != MANIFEST
        and not p.name.startswith(".")
        and not SKIP_DIRS & set(p.relative_to(root).parts)
    )


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_manifest(root, name, license, source):
    files = model_files(root)
    params, dtype = weights_info([f for f in files if f.suffix == ".safetensors"])
    entries = []
    for f in files:
        print(f"hashing {f.relative_to(root)}", file=sys.stderr)
        entries.append(
            {"path": str(f.relative_to(root)), "size": f.stat().st_size, "sha256": sha256(f)}
        )
    return {
        "version": 1,
        "name": name,
        "bytes": sum(e["size"] for e in entries),
        "params": params,
        "dtype": dtype,
        "license": license,
        "source": source,
        "arch": arch_info(json.loads((root / "config.json").read_text())),
        "files": entries,
    }


# --- bucket ---


class Bucket:
    def __init__(self):
        account = os.environ.get("VASTX_R2_ACCOUNT_ID")
        self.env = {
            **os.environ,
            "AWS_ACCESS_KEY_ID": os.environ.get("VASTX_R2_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("VASTX_R2_SECRET_ACCESS_KEY", ""),
            "AWS_REGION": "auto",
        }
        if not (account and self.env["AWS_ACCESS_KEY_ID"] and self.env["AWS_SECRET_ACCESS_KEY"]):
            sys.exit(
                "set VASTX_R2_ACCOUNT_ID, VASTX_R2_ACCESS_KEY_ID and VASTX_R2_SECRET_ACCESS_KEY"
            )
        self.endpoint = f"https://{account}.r2.cloudflarestorage.com"
        self.name = os.environ.get("VASTX_R2_BUCKET", "models")

    def url(self, key=""):
        return f"s3://{self.name}/{key}"

    def s5(self, *args, capture=False, check=True):
        cmd = ["s5cmd", "--endpoint-url", self.endpoint, *args]
        try:
            return subprocess.run(cmd, env=self.env, check=check, capture_output=capture, text=True)
        except FileNotFoundError:
            sys.exit("s5cmd not found; install it with `brew install s5cmd`")
        except subprocess.CalledProcessError as e:
            sys.exit(f"s5cmd failed: {(e.stderr or '').strip() or e}")

    def run_batch(self, commands):
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.write("\n".join(commands) + "\n")
            f.flush()
            self.s5("--numworkers", "8", "run", f.name)

    def keys(self, prefix):
        r = self.s5("--json", "ls", self.url(f"{prefix}*"), capture=True, check=False)
        return [json.loads(line)["key"] for line in r.stdout.splitlines() if line.strip()]

    def put_json(self, key, value, max_age):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(value, f, indent=1)
            f.flush()
            self.s5(
                "cp", "--content-type", "application/json",
                "--cache-control", f"public, max-age={max_age}",
                f.name, self.url(key),
            )

    def catalog(self):
        r = self.s5("cat", self.url(CATALOG), capture=True, check=False)
        if r.returncode != 0:
            if "NoSuchKey" in r.stderr or "not found" in r.stderr.lower():
                return {"version": 1, "models": []}
            sys.exit(f"reading catalog failed: {r.stderr.strip()}")
        return json.loads(r.stdout)


def upsert(catalog, manifest):
    entry = {k: v for k, v in manifest.items() if k not in ("version", "files")}
    models = [m for m in catalog["models"] if m["name"] != entry["name"]] + [entry]
    return {**catalog, "models": sorted(models, key=lambda m: m["name"])}


def remove(catalog, name):
    return {**catalog, "models": [m for m in catalog["models"] if m["name"] != name]}


# --- commands ---


def resolve_source(src, revision):
    """Return (local dir, license, source) for a local path or hf:org/repo."""
    if not src.startswith("hf:"):
        root = Path(src).expanduser().resolve()
        if not (root / "config.json").exists():
            sys.exit(f"{root} has no config.json")
        return root, readme_license(root), None
    from huggingface_hub import HfApi, snapshot_download

    repo = src[3:]
    info = HfApi().model_info(repo, revision=revision)
    sha = info.sha or sys.exit(f"no commit sha for {repo}")
    # One directory per revision, so files from an older revision can't leak in.
    root = Path.home() / ".cache" / "vastx" / "staging" / f"{repo.replace('/', '--')}--{sha[:12]}"
    print(f"downloading {repo}@{sha[:12]} to {root}", file=sys.stderr)
    snapshot_download(repo, revision=sha, local_dir=root, allow_patterns=HF_PATTERNS)
    license = (info.card_data or {}).get("license") or readme_license(root)
    return root, license, {"hf": repo, "revision": sha}


def cmd_add(a):
    root, license, source = resolve_source(a.src, a.revision)
    name = (a.name or (source["hf"].split("/")[-1] if source else root.name)).lower()
    check_name(name)
    license = a.license or license
    if not license:
        sys.exit("no license in README.md frontmatter; pass --license")
    manifest = build_manifest(root, name, license, source)
    if a.dry_run:
        print(json.dumps(manifest, indent=1))
        return
    b = Bucket()
    # Files, then manifest, then catalog: a model is listed only once it is complete.
    # Upload exactly the manifest's files; -s -u skips ones already there unchanged.
    b.run_batch(
        f'cp -s -u -c 8 -p 64 "{root / e["path"]}" "{b.url(f"{name}/{e["path"]}")}"'
        for e in manifest["files"]
    )
    b.put_json(f"{name}/{MANIFEST}", manifest, max_age=60)
    b.put_json(CATALOG, upsert(b.catalog(), manifest), max_age=60)
    keep = {b.url(f"{name}/{p}") for p in [MANIFEST, *(e["path"] for e in manifest["files"])]}
    stale = [k for k in b.keys(f"{name}/") if k not in keep]
    if stale:
        b.run_batch(f'rm "{k}"' for k in stale)
    gb = manifest["bytes"] / 1e9
    print(f"published {name}: {gb:.1f} GB, {manifest['params'] / 1e9:.1f}B params", file=sys.stderr)


def cmd_ls(_):
    for m in Bucket().catalog()["models"]:
        print(
            f"{m['name']:<40} {m['bytes'] / 1e9:>7.1f} GB {m['params'] / 1e9:>7.1f}B "
            f"{m['dtype']:<6} {m['license']}"
        )


def cmd_rm(a):
    check_name(a.name)
    b = Bucket()
    if not a.yes and input(f"delete {a.name} from {b.name}? [y/N] ").strip() not in ("y", "yes"):
        sys.exit("aborted")
    # Unlist first so nobody starts a download of a half-deleted model.
    b.put_json(CATALOG, remove(b.catalog(), a.name), max_age=60)
    b.s5("rm", b.url(f"{a.name}/*"))


def main():
    p = argparse.ArgumentParser(description="Publish models to the vastx catalog bucket")
    sub = p.add_subparsers(required=True)
    add = sub.add_parser("add", help="upload a model and list it in the catalog")
    add.add_argument("src", help="local directory or hf:org/repo")
    add.add_argument("--name", help="defaults to the repo or directory name, lowercased")
    add.add_argument("--revision", help="HF branch, tag or commit (default: main)")
    add.add_argument("--license", help="overrides the license from the model card")
    add.add_argument("--dry-run", action="store_true", help="print the manifest and stop")
    add.set_defaults(fn=cmd_add)
    sub.add_parser("ls", help="list the catalog").set_defaults(fn=cmd_ls)
    rm = sub.add_parser("rm", help="unlist and delete a model")
    rm.add_argument("name")
    rm.add_argument("-y", "--yes", action="store_true")
    rm.set_defaults(fn=cmd_rm)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
