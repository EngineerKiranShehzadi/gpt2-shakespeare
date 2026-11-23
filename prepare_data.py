from pathlib import Path
from urllib.request import urlopen


URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    text_path = DATA_DIR / "shakespeare.txt"

    if text_path.exists():
        print(f"Using existing data: {text_path}")
    else:
        print("Downloading Tiny Shakespeare dataset...")
        try:
            text = urlopen(URL, timeout=30).read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Could not download dataset: {exc}") from exc
        text_path.write_text(text, encoding="utf-8")

    text = text_path.read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    (DATA_DIR / "train.txt").write_text(text[:split], encoding="utf-8")
    (DATA_DIR / "val.txt").write_text(text[split:], encoding="utf-8")

    print(f"Prepared {len(text):,} characters")
    print(f"Train: {len(text[:split]):,} characters")
    print(f"Validation: {len(text[split:]):,} characters")


if __name__ == "__main__":
    main()
