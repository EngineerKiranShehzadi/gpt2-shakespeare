from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    block_size: int = 128
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    n_kv_head: int = 2
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_interval: int = 500
    seed: int = 1337
    cpu_threads: int = 2


MODEL_CONFIG = ModelConfig()
TRAIN_CONFIG = TrainConfig()
