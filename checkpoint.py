import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

FORMAT_VERSION = 1
CHECKPOINT_FIELDS = {
    "model_state", "optimizer_state", "completed_step", "best_validation_loss",
    "best_checkpoint", "model_config", "training_config", "torch_rng_state",
    "tokenizer_path", "tokenizer_sha256", "data_sha256", "validation_loss",
}


def save_checkpoint(path, state):
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(state, temporary_path)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink()


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def require_training_checkpoint(state):
    if (state.get("format_version") != FORMAT_VERSION
            or not CHECKPOINT_FIELDS.issubset(state)
            or not isinstance(state["optimizer_state"], dict)
            or not state["optimizer_state"].get("param_groups")
            or type(state["completed_step"]) is not int
            or not 0 < state["completed_step"] <= state["training_config"]["max_steps"]):
        raise ValueError(
            "Checkpoint lacks supported optimizer or step metadata for an exact resume. "
            "Start a fresh experiment with --run-dir and omit --resume."
        )
