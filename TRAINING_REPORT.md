# Shakespeare GPT Training Report

## 5,000-Step Result

- Starting checkpoint: `checkpoints/shakespeare-gpt-final.pt`
- Final checkpoint: `checkpoints/shakespeare-gpt-5000.pt`
- Final training loss: `3.7611`
- Authoritative deterministic validation loss: `3.9200`
- Resumed training time: `495.80 seconds` (8.26 minutes)

## Validation Method

Validation uses 10 fixed, evenly spaced windows from `data/val.txt`. Each window contains 127 target tokens at the configured block size. The reported value is the mean cross-entropy loss across these windows, making repeated evaluations reproducible.

Random-batch validation estimates are excluded from this report because they are not reproducible.

## Checkpoints Preserved

- `checkpoints/shakespeare-gpt.pt`
- `checkpoints/shakespeare-gpt-step100-backup.pt`
- `checkpoints/shakespeare-gpt-final.pt`
- `checkpoints/shakespeare-gpt-step1000-backup.pt`
- `checkpoints/shakespeare-gpt-5000.pt`

The GQA, BPE tokenizer, and model architecture were unchanged.
