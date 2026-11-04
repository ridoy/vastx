# vastx

A CLI for renting cloud GPUs (vast.ai first, other providers later) for LLM training and inference.

The first feature is data loading.
You push model weights from your machine to a Cloudflare R2 bucket once, then pull them onto any rented instance with many parallel ranged downloads.
That's far faster than `scp` from a home connection, and you can bring up as many instances as you want from the same upload.

## Install

```sh
make install            # builds into ./bin and copies to ~/.local/bin
```

This installs `vastx` for your machine plus static `vastx-linux-amd64` and `vastx-linux-arm64` next to it.
`vastx share` uploads the Linux binary so a fresh instance can bootstrap itself with one `curl`.

## Setup

Create an R2 bucket, then an API token under R2 > Manage API tokens with "Object Read & Write" scoped to that bucket.

```sh
vastx config set account_id <cloudflare account id>
vastx config set access_key_id <token access key id>
vastx config set secret_access_key <token secret>
vastx config set bucket <bucket>
vastx config show
```

Settings are stored in `~/.config/vastx/config.json` (mode 0600).
Each one can also come from an environment variable (`VASTX_R2_ACCOUNT_ID`, `VASTX_R2_ACCESS_KEY_ID`, `VASTX_R2_SECRET_ACCESS_KEY`, `VASTX_R2_BUCKET`), which takes precedence.
Any other S3-compatible store works by setting `endpoint` (and `region`) instead of `account_id`.

## Usage

```sh
# On your machine: upload a model directory (or a single file).
vastx push ~/models/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct

# Create a credential-free link plus a bootstrap command for the instance.
vastx share meta-llama/Llama-3.1-8B-Instruct
```

`share` prints something like:

```sh
curl -fsSL -o /usr/local/bin/vastx '<presigned binary url>' && chmod +x /usr/local/bin/vastx && \
  vastx pull '<presigned share url>' '/workspace/models/Llama-3.1-8B-Instruct'
```

Paste that into the instance's shell, or put it in a vast.ai onstart script (`vastx share <name> -o cmd` prints only the command).
The instance never sees your R2 keys, which matters because rented hosts are run by third parties.
Links expire after `--ttl` (default 24h, max 7 days).

Other commands:

```sh
vastx pull <name> [dest]     # pull by name using your credentials
vastx ls                     # list models in the bucket
vastx rm <name>              # delete a model
```

## How it works

Every file is split into fixed-size chunks (64 MiB by default) and the manifest records a SHA-256 per chunk.

`push` reads chunks with a worker pool and uploads them as multipart parts, hashing each one on the way.
Workers pull from one queue spanning all files, so a single 70 GB shard and a thousand small files both keep every connection busy.
Files whose size and mtime match the previous push are skipped, and files deleted locally are removed from the bucket.
The manifest is checkpointed while the push runs, so an interrupted push resumes at file granularity.
Memory use is concurrency times chunk size (512 MiB by default).

`pull` issues one ranged GET per chunk, 32 at a time by default, over separate HTTP/1.1 connections.
A single connection to R2 tops out far below a datacenter NIC, and HTTP/2 would multiplex everything over one TCP connection, so parallel HTTP/1.1 is what gets the bandwidth.
Each response streams straight into its offset in a preallocated file while being hashed, so nothing is buffered beyond a few MiB per worker and there's no separate verification pass.
Failed, stalled (no data for 60s) or corrupted chunks are retried with backoff.
Files are written as `<file>.vastx-partial` and renamed when complete.
Running the same pull again after an interruption verifies what's on disk and downloads only the chunks that are missing or wrong.
It also checks free disk space before starting.

Hugging Face cache snapshots work as push sources since symlinks are followed.
`.git`, `.cache/huggingface` and `.DS_Store` are skipped; add more with `--exclude`.

### Bucket layout

```
<name>/<file path>                 model files, as-is (usable by other tools)
<name>/.vastx/manifest.json        manifest: files, sizes, chunk hashes
.vastx/shares/<name>-<id>.json     share manifests with presigned URLs
.vastx/bin/<hash>/vastx-linux-*    binaries for instance bootstrap
```

Model names may contain slashes but can't nest inside one another.

### Tuning

- `vastx pull -j 64` for very fast instance networks; watch the reported rate and raise it until it stops improving.
- `vastx push -j 16` if your upload link has headroom and you have the memory.
- `--chunk-size` changes the part size for new pushes; files above 625 GiB get larger chunks automatically to stay under R2's 10,000-part limit.

## Development

```sh
make test   # go test -race ./...
make lint   # go vet + gofmt
```

The end-to-end test in `internal/cli` runs the real commands against an in-process fake S3 server (gofakes3), covering push, re-push, pull by name and by share link, corruption repair, resume, deletion, and the bootstrap binary URL.
