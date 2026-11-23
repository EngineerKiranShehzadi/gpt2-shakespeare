import math

import torch
import torch.nn as nn



class CausalSelfAttention(nn.Module):
    """Multi-head attention with fewer key/value heads (grouped-query attention)."""

    def __init__(self, config):
        super().__init__()
        if config.n_head % config.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        if config.n_embd != self.n_head * self.head_dim:
            raise ValueError("n_embd must be divisible by n_head")

        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        kv_size = self.n_kv_head * self.head_dim
        self.key = nn.Linear(config.n_embd, kv_size, bias=False)
        self.value = nn.Linear(config.n_embd, kv_size, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        q = self.query(x).view(batch_size, sequence_length, self.n_head, self.head_dim)
        k = self.key(x).view(batch_size, sequence_length, self.n_kv_head, self.head_dim)
        v = self.value(x).view(batch_size, sequence_length, self.n_kv_head, self.head_dim)

        repeat_count = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat_count, dim=2)
        v = v.repeat_interleave(repeat_count, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.residual_dropout(self.output(output))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.language_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, token_ids, targets=None):
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError("Sequence is longer than block_size")

        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.language_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (
            type(top_k) is not int or not 1 <= top_k <= self.config.vocab_size
        ):
            raise ValueError("top_k must be an integer between 1 and vocab_size")
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            if top_k is not None and top_k < self.config.vocab_size:
                values, indices = next_logits.topk(top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(-1, indices, values)
            next_logits = (next_logits - next_logits.amax(dim=-1, keepdim=True)) / temperature
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
