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
vastx push ~/models/llama-3.1-8b        # mirror a directory; unchanged files are skipped
vastx share llama-3.1-8b                # prints a command to run on the instance
vastx pull llama-3.1-8b [dest]          # pull with your own credentials
vastx ls
vastx rm llama-3.1-8b
```

`share` mints temporary R2 credentials that are read-only, limited to the model's prefix, and expire after `--hours` (default 24, max 168).
It prints one line that installs s5cmd if it's missing and syncs the model into `/workspace/models/<name>`.
Paste it into the instance shell or a vast.ai onstart script, so your real keys never touch a rented machine.

Transfers are parallel across files (`-w`) and across parts within a file (`-c`, `-p` MiB).
Defaults are conservative for push (memory use is roughly workers x concurrency x part size) and aggressive for pull.
Re-running pull resumes at file granularity.
