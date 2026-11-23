import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import torch

from checkpoint import FORMAT_VERSION, read_checkpoint, require_training_checkpoint, save_checkpoint
from config import MODEL_CONFIG, TRAIN_CONFIG, ModelConfig, TrainConfig
from model import GPT
from tokenizer import BYTE_TOKENIZER_PATH, DATA_PATH, file_sha256, load_tokenizer

VAL_PATH = DATA_PATH.with_name("val.txt")


def get_batch(data, block_size, batch_size, device):
    if len(data) <= block_size:
        raise ValueError("Not enough data for a training batch")
    starts = torch.randint(0, len(data) - block_size, (batch_size,), device="cpu")
    sequences = torch.stack([data[start : start + block_size] for start in starts])
    return sequences[:, :-1].to(device), sequences[:, 1:].to(device)


@torch.no_grad()
def estimate_validation_loss(model, data, device, eval_batches=10):
    block_size = model.config.block_size
    available_starts = len(data) - block_size
    if available_starts <= 0:
        raise ValueError("Not enough validation data")
    was_training = model.training
    model.eval()
    try:
        losses = []
        for batch_index in range(eval_batches):
            start = batch_index * available_starts // eval_batches
            sequence = data[start : start + block_size].unsqueeze(0).to(device)
            _, loss = model(sequence[:, :-1], sequence[:, 1:])
            losses.append(loss.item())
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def learning_rate_at(step, config):
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_start = max(config.warmup_steps, 1)
    decay_steps = max(config.max_steps - decay_start, 1)
    progress = (step - decay_start) / decay_steps
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def validate_config(config):
    if min(config.max_steps, config.eval_interval, config.cpu_threads, config.batch_size) < 1:
        raise ValueError("Steps, evaluation interval, threads, and batch size must be positive")
    if not 0 <= config.warmup_steps < config.max_steps:
        raise ValueError("Warmup steps must be nonnegative and below max steps")
    if (not math.isfinite(config.learning_rate) or config.learning_rate <= 0
            or not 0 <= config.min_learning_rate <= config.learning_rate):
        raise ValueError("Learning rates must be finite with 0 <= minimum <= peak and peak > 0")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Shakespeare GPT experiment on CPU.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="New directory for a fresh experiment")
    source.add_argument("--resume", type=Path, help="Resume the latest checkpoint in its run directory")
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer file (must match the saved hash when resuming)")
    parser.add_argument("--stop-after", type=int, help="Pause after this absolute step without changing the schedule")
    for name, value in asdict(TRAIN_CONFIG).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value))
    return parser.parse_args()


def run(args):
    overrides = {
        key: getattr(args, key) for key in asdict(TRAIN_CONFIG)
        if getattr(args, key) is not None
    }
    state = None
    start_step = 0
    best_loss = math.inf
    best_checkpoint = None
    if args.resume:
        state = read_checkpoint(args.resume)
        require_training_checkpoint(state)
        if overrides:
            raise ValueError("Resume uses the saved training configuration, including the schedule horizon")
        model_config = ModelConfig(**state["model_config"])
        config = TrainConfig(**state["training_config"])
        start_step = state["completed_step"]
        best_loss = state["best_validation_loss"]
        best_checkpoint = state["best_checkpoint"]
        run_dir = args.resume.resolve().parent
        tokenizer_path = args.tokenizer or Path(state["tokenizer_path"])
        if (run_dir / "checkpoint-final.pt").exists() or any(
            int(path.stem.rsplit("-", 1)[-1]) > start_step
            for path in run_dir.glob("checkpoint-*-*.pt")
        ):
            raise ValueError("Run already has a later checkpoint or is complete")
    else:
        model_config = MODEL_CONFIG
        config = replace(TRAIN_CONFIG, **overrides)
        run_dir = args.run_dir.resolve()
        tokenizer_path = args.tokenizer or BYTE_TOKENIZER_PATH

    validate_config(config)
    stop_step = config.max_steps if args.stop_after is None else args.stop_after
    if not start_step < stop_step <= config.max_steps:
        raise ValueError("Stop step must be after the saved step and at or below max steps")

    tokenizer_path = tokenizer_path.resolve()
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("Tokenizer vocabulary size differs from the model configuration")
    tokenizer_hash = file_sha256(tokenizer_path)
    data_hashes = {path.name: file_sha256(path) for path in (DATA_PATH, VAL_PATH)}
    if state and (state["tokenizer_sha256"] != tokenizer_hash or state["data_sha256"] != data_hashes):
        raise ValueError("Tokenizer or Shakespeare split contents differ from the checkpoint")
    token_ids, val_ids = (
        torch.tensor(tokenizer.encode(path.read_text(encoding="utf-8")).ids, dtype=torch.long)
        for path in (DATA_PATH, VAL_PATH)
    )

    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    device = torch.device("cpu")
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if state:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["torch_rng_state"])
        print(f"Resuming after step {start_step} from {args.resume}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        print(f"Fresh experiment in {run_dir}")
    print(f"CPU threads: {config.cpu_threads} | Parameters: {model.parameter_count():,}")

    metadata = {
        "format_version": FORMAT_VERSION,
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_hash,
        "data_sha256": data_hashes,
    }
    model.train()
    for step in range(start_step + 1, stop_step + 1):
        lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = get_batch(token_ids, model_config.block_size, config.batch_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        evaluate = step % config.eval_interval == 0 or step == config.max_steps
        if not evaluate and step != stop_step:
            continue
        training_loss = loss.item()
        validation_loss = estimate_validation_loss(model, val_ids, device) if evaluate else None
        improved = evaluate and validation_loss < best_loss
        checkpoint_path = run_dir / f"checkpoint-step-{step:06d}.pt"
        if improved:
            best_loss = validation_loss
            best_checkpoint = f"checkpoint-best-step-{step:06d}.pt"
        checkpoint_state = {
            **metadata,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "completed_step": step,
            "best_validation_loss": best_loss,
            "best_checkpoint": best_checkpoint,
            "validation_loss": validation_loss,
            "training_loss": training_loss,
            "learning_rate": lr,
        }
        if improved:
            checkpoint_path = run_dir / best_checkpoint
            save_checkpoint(checkpoint_path, checkpoint_state)
        if step == config.max_steps:
            checkpoint_path = run_dir / "checkpoint-final.pt"
        if not improved or step == config.max_steps:
            save_checkpoint(checkpoint_path, checkpoint_state)
        validation_text = f"{validation_loss:.4f}" if evaluate else "scheduled for a later step"
        print(f"Step {step}/{config.max_steps} | lr {lr:.6g} | train {training_loss:.4f} | val {validation_text}")
        print(f"Saved {checkpoint_path}")

    if best_checkpoint:
        print(f"Best validation loss {best_loss:.4f}: {run_dir / best_checkpoint}")
    if stop_step == config.max_steps:
        report = {
            **metadata,
            "completed_step": stop_step,
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint": str(run_dir / best_checkpoint) if best_checkpoint else None,
            "best_validation_loss": best_loss if best_checkpoint else None,
            "validation_loss": validation_loss,
            "training_loss": training_loss,
        }
        with (run_dir / "run-report.json").open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
