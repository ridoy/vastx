# vastx

Tooling for renting cloud GPUs (vast.ai first) for LLM training and inference.
Right now it loads model weights onto instances through a Cloudflare R2 bucket, using [s5cmd](https://github.com/peak/s5cmd) for the transfers.

## Setup

```sh
brew install s5cmd
cargo install --path .

export VASTX_R2_ACCOUNT_ID=...          # Cloudflare account ID
export VASTX_R2_ACCESS_KEY_ID=...       # R2 API token with Object Read & Write on the bucket
export VASTX_R2_SECRET_ACCESS_KEY=...
export VASTX_R2_BUCKET=...
export CLOUDFLARE_API_TOKEN=...         # only for `share`; needs R2 write permission
```

`VASTX_S3_ENDPOINT` points vastx at any other S3-compatible store.

## Usage

```sh
vastx share llama-3.1-8b                # prints a command to run on the instance
vastx pull llama-3.1-8b [dest]          # pull with your own credentials
```

`share` mints temporary R2 credentials that are read-only, limited to the model's prefix, and expire after `--hours` (default 24, max 168).
It prints one line that installs s5cmd if it's missing and syncs the model into `/workspace/models/<name>`.
Paste it into the instance shell or a vast.ai onstart script, so your real keys never touch a rented machine.

Transfers are parallel across files (`-w`) and across parts within a file (`-c`, `-p` MiB).
Re-running pull resumes at file granularity.

## Publishing models (maintainer)

`scripts/publish.py` uploads models to the catalog bucket.
It needs s5cmd, the `VASTX_R2_*` variables above (`VASTX_R2_BUCKET` defaults to `models`), and either [uv](https://docs.astral.sh/uv/) or Python 3.11+ with `huggingface_hub` installed.

```sh
scripts/publish.py add hf:Qwen/Qwen2.5-7B-Instruct           # download from Hugging Face, then publish
scripts/publish.py add ./my-model --name my-model --license mit
scripts/publish.py add hf:org/repo --dry-run                  # print the manifest without uploading
scripts/publish.py ls
scripts/publish.py rm my-model
```

Each model gets `<name>/vastx.json` with its files, sizes, sha256 hashes, parameter count, dtype, license, upstream revision, and the architecture numbers needed to size GPUs.
`catalog.json` at the bucket root lists every model without the file lists.
Files upload first, then the manifest, then the catalog, so a model only shows up once it is complete.
`rm` unlists a model before deleting its files.
Hugging Face downloads are staged under `~/.cache/vastx/staging/`; for large models, run the script on a cloud machine so the weights don't go through your home connection.
