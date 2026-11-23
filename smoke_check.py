import torch

from config import MODEL_CONFIG
from model import GPT
from tokenizer import load_tokenizer


def main():
    torch.manual_seed(1337)
    tokenizer = load_tokenizer()

    text = "To be, or not to be, that is the question."
    ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.long)
    inputs = ids[:, :-1]
    targets = ids[:, 1:]

    model = GPT(MODEL_CONFIG)
    logits, loss = model(inputs, targets)

    print(f"Python/PyTorch smoke check")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model parameters: {model.parameter_count():,}")
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size()}")
    print(f"Token count: {ids.size(1)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.4f}")

    model.generate(ids[:, :8], max_new_tokens=8, temperature=0.8)
    print("Smoke check passed: forward pass and generation completed.")


if __name__ == "__main__":
    main()
