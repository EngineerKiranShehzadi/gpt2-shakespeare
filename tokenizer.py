import argparse
import hashlib
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "train.txt"
TOKENIZER_PATH = PROJECT_DIR / "data" / "shakespeare-bpe.json"
BYTE_TOKENIZER_PATH = PROJECT_DIR / "data" / "shakespeare-bpe-full-byte.json"
VOCAB_SIZE = 1000
SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>"]

CHECK_SAMPLE = "FIRST CITIZEN:\n  To be, or not to be?\t'Tis well!\r\n— “café”\u00a0漢字 🦉\u200d✨"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_tokenizer(output_path=BYTE_TOKENIZER_PATH, data_path=DATA_PATH):
    if not data_path.exists():
        raise FileNotFoundError("Run prepare_data.py first")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    alphabet_method = getattr(ByteLevel, "alphabet", None)
    if not callable(alphabet_method):
        print("ByteLevel.alphabet() unavailable; full byte coverage is unverified.")
    alphabet = {"initial_alphabet": alphabet_method()} if callable(alphabet_method) else {}
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=SPECIAL_TOKENS, **alphabet
    )
    tokenizer.train([str(data_path)], trainer)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(tokenizer.to_str(pretty=True))
    print(f"Saved BPE tokenizer: {output_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


def load_tokenizer(path=TOKENIZER_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer missing: {path}. Run tokenizer.py train first.")
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def check_tokenizer(tokenizer):
    ids = tokenizer.encode(CHECK_SAMPLE).ids
    alphabet_method = getattr(ByteLevel, "alphabet", None)
    vocab = tokenizer.get_vocab()
    missing_bytes = (
        len(set(alphabet_method()).difference(vocab)) if callable(alphabet_method) else None
    )
    return {
        "vocabulary_size": len(vocab),
        "unknown_token_count": ids.count(tokenizer.token_to_id("<unk>")),
        "token_count": len(ids),
        "round_trip": tokenizer.decode(ids) == CHECK_SAMPLE,
        "missing_byte_symbols": missing_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train or check the 1,000-token byte-level BPE.")
    parser.add_argument("command", choices=["train", "check"], nargs="?", default="train")
    parser.add_argument("--tokenizer", type=Path, default=BYTE_TOKENIZER_PATH)
    args = parser.parse_args()
    if args.command == "train":
        train_tokenizer(args.tokenizer)
    else:
        result = check_tokenizer(load_tokenizer(args.tokenizer))
        for key, value in result.items():
            print(f"{key}: {value}")
        if (not result["round_trip"] or result["unknown_token_count"]
                or result["missing_byte_symbols"] != 0):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
