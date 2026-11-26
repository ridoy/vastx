# vastx

vastx makes it easy to run open models if you don't have a GPU. There are some other benefits but this is mostly just a recreational exercise for me.

## Setup

```sh
git clone https://github.com/ridoy/vastx.git
cd vastx
cargo install --path .
```

`VASTX_S3_ENDPOINT` points vastx at any other S3-compatible store.

## Usage

First time: just run `vastx`.
1. Logs you in to vast.ai
2. Prompts you to select a model (as of now, just qwen2.5-1.5b, qwen3-8b, mistral-7b or deepseek-r1-7b`)
3. Pick a GPU from a couple rightsized options.
4. Wait 3-4 minutes for the instance to start, download the model, install llama.cpp, start serving...
5. You are now live. Prompt your chatbot through cURL! (at this point an example command will be printed)

Some other commands:
```
vastx ps             # lists running instances
vastx connect <id>   # connects to instance with id <id>
vast down <id>       # destroys instance so you stop getting billed
```

# What's next?

- [ ] Better chat UI. Should basically open a shell that lets you send messages. cURLing it is a crappy UX
- [ ] More models to select. E.g. providing a HuggingFace URL, object hosted on S3 or R2, local finetuned models...
- [ ] Multi-GPU setups. Need to research this more, but in principle one should be able to split layers across machines to enable larger models.
- [ ] Make it available on Homebrew. Because I want this readme to say "`brew install vastx && vastx` is all you need to run {Qwen, Kimi, Deepseek, etc...}". Or maybe "Chat with an open model in 3 minutes with `brew install vastx && vastx`"

