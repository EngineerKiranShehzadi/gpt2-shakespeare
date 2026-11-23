# GPT-2-style Shakespeare model

A CPU-first autoregressive decoder trained on Tiny Shakespeare, inspired by [Karpathy's GPT-2 code-along](https://www.youtube.com/watch?v=l8pRSuU81PU). The baseline has 998,144 parameters, 4 Transformer blocks, a 128-token context, 4 query heads sharing 2 key/value heads, and a 1,000-token byte-level BPE vocabulary. The architecture and custom GQA are shared by the original and improved experiments.

## Setup and tokenizer

Run these commands in `gpt2-shakespeare` using PowerShell. With the project environment already installed, start at the tokenizer check.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe tokenizer.py train
.\.venv\Scripts\python.exe tokenizer.py check
```

Data preparation splits the text 90/10 by character. Tokenizer training uses only `data/train.txt` and writes `data/shakespeare-bpe-full-byte.json`. The trainer keeps the four original special tokens and seeds BPE with the 256 byte symbols from `ByteLevel.alphabet()` when available. The check reports vocabulary size, token count, unknown-token count and missing byte symbols, and verifies a round-trip for Shakespeare text with punctuation, whitespace and Unicode. The trainer refuses to replace an existing tokenizer file.

The original `data/shakespeare-bpe.json`, `checkpoints/` and `samples.txt` remain paired baseline artifacts. Even at the same vocabulary size, separately trained tokenizers can assign different IDs. New checkpoints record a tokenizer SHA-256 and verify it before resuming or generating. Any future vocabulary-size experiment needs its own tokenizer, matching model configuration, and run directory.

## Train, pause and resume

This short CPU example trains 12 total updates. The first command pauses after update 5. The second finishes using the original 12-step schedule.

```powershell
.\.venv\Scripts\python.exe train.py --run-dir runs/my-cpu-smoke --max-steps 12 --warmup-steps 3 --eval-interval 4 --stop-after 5
.\.venv\Scripts\python.exe train.py --resume runs/my-cpu-smoke/checkpoint-step-000005.pt
```

For a full fresh baseline experiment, run `train.py --run-dir runs/my-full-run`. Defaults are 5,000 updates, batch size 8, two CPU threads, seed 1337, 100 warmup steps and evaluation every 500 steps. The learning rate ramps to 0.0003 and decays by cosine to 0.00003. Use `train.py --help` for training options. A full CPU run takes substantially longer than the smoke run. The historical 4,000-update continuation took about eight minutes on the machine used for that run. Current runtime varies by CPU.

- Each fresh experiment requires a new directory. Scheduled evaluations use 10 fixed validation windows and log the step, learning rate, training loss and validation loss.
- `checkpoint-best-step-*.pt` records each new validation minimum. Regular progress and planned pauses use `checkpoint-step-*.pt`. `checkpoint-final.pt` is saved separately at the configured final step. The terminal and `run-report.json` identify the best checkpoint and its loss.
- Resume from the highest saved step in the same run directory. The checkpoint restores model weights, AdamW state and the CPU PyTorch RNG used by dropout and batch selection. It stores the completed step and configuration, and compares tokenizer and split hashes. The schedule is calculated from the saved configuration. `--stop-after` can pause at another step without changing that schedule. An unexpected interruption can lose progress since the last checkpoint.
- The historical weights-only files lack optimizer and reliable step metadata for exact resume. Start a fresh run for the full-byte tokenizer. Exact continuation was verified on the same CPU/PyTorch setup. Results can differ across environments.
- The final report stores both configurations, seed, tokenizer and checkpoint paths, hashes, step count and losses. Validation is a small fixed-window estimate. A lower loss alone does not guarantee fluent continuations, and losses across different tokenizers are not directly comparable.

## Generate and compare samples

```powershell
.\.venv\Scripts\python.exe generate.py --checkpoint runs/my-cpu-smoke/checkpoint-final.pt --output runs/my-cpu-smoke/samples-temperature.txt
.\.venv\Scripts\python.exe generate.py --checkpoint runs/my-cpu-smoke/checkpoint-final.pt --top-k 40 --output runs/my-cpu-smoke/samples-top-k-40.txt
```

Both commands use the same three prompts and per-prompt seeds. Defaults are temperature 0.8, seed base 1337 and 100 new tokens. `--top-k` restricts each sampling step to the given number of highest-scoring tokens. It changes generation only. Any `top_k` from 1 through 1,000 is supported, and omitting it samples from the full vocabulary. Pass a new output path for each comparison. Running `generate.py` without arguments uses the paired legacy 5,000-step checkpoint and writes `samples-improved.txt`. Missing or mismatched artifacts fail instead of using random weights.

If you use a saved run from a different machine, pass `--tokenizer data/shakespeare-bpe-full-byte.json` to generation or resume. The checkpoint verifies the file's contents before use.

## Checks and results

```powershell
.\.venv\Scripts\python.exe smoke_check.py
.\.venv\Scripts\python.exe -m unittest -v
```

The verified 12-step run is in `runs/gqa-bpe-cpu-smoke/`. Its best and final validation loss is 6.8293. The resulting samples demonstrate the pipeline on a short training run. The original 5,000-step samples and their [training report](TRAINING_REPORT.md) provide the historical baseline.
