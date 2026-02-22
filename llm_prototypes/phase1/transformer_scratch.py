"""
Phase 1.1 — Transformer from Scratch
=====================================
Implements scaled dot-product attention, multi-head attention,
positional encoding, and a full transformer decoder, then trains
a character-level language model on Shakespeare.

Run:
    python phase1/transformer_scratch.py

Requirements: torch, numpy, rich
"""

import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from rich import print
    from rich.console import Console
    from rich.progress import track
    console = Console()
except ImportError:
    console = None
    def track(it, description=""):
        return it


# ── 1. Scaled Dot-Product Attention ──────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Paper: "Attention Is All You Need", Vaswani et al. 2017, Section 3.2.1
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,   # (B, heads, T, d_k)
        K: torch.Tensor,   # (B, heads, T, d_k)
        V: torch.Tensor,   # (B, heads, T, d_v)
        mask: Optional[torch.Tensor] = None,  # (B, 1, T, T) causal mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        # Scores: (B, heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


# ── 2. Multi-Head Attention ───────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    Paper: Section 3.2.2
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, heads, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = Q.size(0)

        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        attn_out, attn_weights = self.attention(Q, K, V, mask)

        # Concat heads: (B, heads, T, d_k) → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_O(attn_out), attn_weights


# ── 3. Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Paper: Section 3.5
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── 4. Feed-Forward Network ───────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Position-wise FFN: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 5. Transformer Decoder Block ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Decoder-only block (GPT-style):
    x → LayerNorm → MaskedMHA → residual → LayerNorm → FFN → residual
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm residual (more stable than post-norm)
        normed = self.norm1(x)
        attn_out, attn_weights = self.mha(normed, normed, normed, mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights


# ── 6. Full Transformer Language Model ───────────────────────────────────────

@dataclass
class TransformerConfig:
    vocab_size: int = 65          # Shakespeare character vocab
    block_size: int = 256         # context length
    d_model: int = 256            # embedding dimension
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1


class TransformerLM(nn.Module):
    """Minimal GPT-style decoder-only transformer for character LM."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.block_size, config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: input embedding = output projection
        self.token_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask of 0s so position i cannot attend to j > i."""
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        return mask

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        x = self.pos_enc(self.token_emb(idx))
        mask = self._causal_mask(T, idx.device)
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, mask)
            attn_maps.append(attn)
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── 7. Shakespeare Dataset ────────────────────────────────────────────────────

class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi.get(c, 0) for c in s], dtype=torch.long)

    def decode(self, t: torch.Tensor) -> str:
        return "".join(self.itos[i.item()] for i in t)


def get_shakespeare() -> str:
    path = "/tmp/shakespeare.txt"
    if not os.path.exists(path):
        print("Downloading Shakespeare corpus...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return f.read()


# ── 8. Training ───────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[bold]Phase 1.1 — Transformer from Scratch[/bold]")
    print(f"Device: {device}\n")

    # Data
    text = get_shakespeare()
    split = int(0.9 * len(text))
    train_text, val_text = text[:split], text[split:]

    config = TransformerConfig()
    train_ds = CharDataset(train_text, config.block_size)
    val_ds = CharDataset(val_text, config.block_size)
    config.vocab_size = train_ds.vocab_size

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model
    model = TransformerLM(config).to(device)
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Block size: {config.block_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5, eta_min=1e-5
    )

    # Training loop (5 epochs on first 200k chars to keep demo fast)
    max_steps = 500          # ~5 minutes on CPU
    eval_interval = 100
    step = 0
    best_val_loss = float("inf")

    model.train()
    train_iter = iter(train_loader)

    print(f"\nTraining for {max_steps} steps...\n")
    losses = []

    for step in range(1, max_steps + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % eval_interval == 0:
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xv, yv in list(val_loader)[:20]:
                    xv, yv = xv.to(device), yv.to(device)
                    _, vl = model(xv, yv)
                    val_losses.append(vl.item())
            val_loss = np.mean(val_losses)
            train_loss = np.mean(losses[-eval_interval:])
            perplexity = math.exp(val_loss)
            print(
                f"Step {step:4d} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} | perplexity={perplexity:.1f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "/tmp/transformer_lm.pt")
            model.train()

        if step % (max_steps // 5) == 0:
            scheduler.step()

    # Generate text
    print("\n[bold]--- Generated Text Sample ---[/bold]")
    model.eval()
    context = train_ds.encode("ROMEO:").unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens=300, temperature=0.8)
    print(train_ds.decode(generated[0]))

    # Visualise attention patterns
    print("\n[bold]--- Attention Pattern Analysis ---[/bold]")
    with torch.no_grad():
        sample_text = "To be or not to be, that is the question"
        tokens = train_ds.encode(sample_text).unsqueeze(0).to(device)
        logits, _ = model(tokens)
        # Get attention from last block
        model.blocks[-1].eval()
        _, attn = model.blocks[-1](
            model.pos_enc(model.token_emb(tokens)),
            model._causal_mask(tokens.size(1), device),
        )
        avg_attn = attn[0].mean(0)  # average over heads: (T, T)
        print(f"Sample: '{sample_text}'")
        print(f"Attention matrix shape: {avg_attn.shape}")
        print(f"Avg attention entropy: {-(avg_attn * (avg_attn + 1e-9).log()).sum(-1).mean():.3f}")

    print(f"\n[bold green]Training complete![/bold green]")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {math.exp(best_val_loss):.1f}")


# ── 9. Attention Visualisation Demo ──────────────────────────────────────────

def demo_attention_mechanics():
    """Demonstrate attention mechanics on a tiny example."""
    print("\n[bold]--- Attention Mechanics Demo ---[/bold]")
    torch.manual_seed(42)

    B, T, d_model, heads = 1, 5, 16, 2
    attn_layer = MultiHeadAttention(d_model, heads)
    pos_enc = PositionalEncoding(d_model, max_len=100)

    # Random token embeddings
    x = torch.randn(B, T, d_model)
    x_with_pe = pos_enc(x)

    # Causal mask
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    out, weights = attn_layer(x_with_pe, x_with_pe, x_with_pe, mask)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention pattern (head 0, causal masked):")
    w = weights[0, 0].detach().numpy()
    for i, row in enumerate(w):
        bar = "".join("█" if v > 0.3 else "▒" if v > 0.1 else "░" for v in row)
        print(f"  token {i}: {bar}  (sum={row.sum():.2f})")


if __name__ == "__main__":
    demo_attention_mechanics()
    train()
