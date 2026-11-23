import argparse
from pathlib import Path
import sys

import torch

from checkpoint import FORMAT_VERSION, read_checkpoint
from config import MODEL_CONFIG, ModelConfig
from model import GPT
from tokenizer import PROJECT_DIR, TOKENIZER_PATH, file_sha256, load_tokenizer

LEGACY_CHECKPOINT = PROJECT_DIR / "checkpoints" / "shakespeare-gpt-5000.pt"
LEGACY_WEIGHTS_SHA256 = "3f21635f7354eb487db7d4d2351237de605bafe09ec6c6b4f1fa759a95eae418"
LEGACY_TOKENIZER_SHA256 = "120918cb3880917e0a64cfe77369640c59fea630bbc4ef0cd30d7bf8325d508d"
PROMPTS = [
    "To be, or not to be",
    "My lord, the king",
    "What wouldst thou do",
]


def main():
    parser = argparse.ArgumentParser(description="Generate continuations from a Shakespeare checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=LEGACY_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "samples-improved.txt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new filename.")

    torch.set_num_threads(2)
    state = read_checkpoint(args.checkpoint)
    if state.get("format_version") == FORMAT_VERSION:
        model_config = ModelConfig(**state["model_config"])
        weights = state["model_state"]
        tokenizer_path = args.tokenizer or Path(state["tokenizer_path"])
        expected_hash = state["tokenizer_sha256"]
    else:
        if file_sha256(args.checkpoint) != LEGACY_WEIGHTS_SHA256:
            parser.error("This weights-only checkpoint has no verified tokenizer pairing.")
        model_config = MODEL_CONFIG
        weights = state
        tokenizer_path = args.tokenizer or TOKENIZER_PATH
        expected_hash = LEGACY_TOKENIZER_SHA256

    if file_sha256(tokenizer_path) != expected_hash:
        parser.error("Tokenizer contents differ from the tokenizer paired with this checkpoint.")
    tokenizer = load_tokenizer(tokenizer_path)
    model = GPT(model_config)
    model.load_state_dict(weights)

    sections = [
        f"Checkpoint: {args.checkpoint.resolve()}",
        f"Tokenizer: {tokenizer_path.resolve()}",
        f"Temperature: {args.temperature}, top_k: {args.top_k}, new tokens: {args.max_new_tokens}",
    ]
    for index, prompt in enumerate(PROMPTS, start=1):
        seed = args.seed + index
        torch.manual_seed(seed)
        prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        try:
            generated_ids = model.generate(
                prompt_ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
            )
        except ValueError as exc:
            parser.error(str(exc))
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        sections.append(f"\nSample {index} (seed {seed})\nPrompt: {prompt}\nGenerated:\n{generated_text}")

    content = "\n".join(sections)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(content + "\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    print(content)
    print(f"Saved samples: {args.output}")


if __name__ == "__main__":
    main()
