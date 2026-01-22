"""
Transformer Model for Generating Stellar Systems

Decoder-only transformer (GPT-style) that generates orbital parameters
for planets in a multi-planet system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)"""

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs.cos(), freqs.sin()


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            cos, sin: Rotary embeddings
            mask: Causal mask
        """
        batch, seq_len, dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, seq_len, dim)
        out = self.out_proj(out)

        return out

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings"""
        seq_len = x.shape[-2]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]

        return torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1
        )


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer decoder block"""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x), cos, sin, mask)
        x = x + attn_out

        # Feed-forward with residual
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out

        return x


class TransformerForGeneration(nn.Module):
    """Decoder-only transformer for generating stellar systems"""

    def __init__(
        self,
        vocab_size: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        head_dim = embedding_dim // num_heads
        self.pos_embedding = RotaryPositionalEmbedding(head_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Layer norm + output projection
        self.ln = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = self.token_embedding(input_ids)

        # Positional embedding
        cos, sin = self.pos_embedding(seq_len, x.device)

        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin, mask)

        # Output projection
        x = self.ln(x)
        logits = self.output_proj(x)

        return logits

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively

        Args:
            prompt: (batch, seq_len) starting tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more diverse)
            top_k: Keep only top-k highest probability tokens
            device: Device to generate on

        Returns:
            generated: (batch, seq_len + max_new_tokens) token sequence
        """
        if device is None:
            device = next(self.parameters()).device

        generated = prompt.clone().to(device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass on most recent tokens (limit to max_seq_len)
                input_seq = generated[:, -self.max_seq_len:]
                logits = self.forward(input_seq)

                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    top_k_values, top_k_indices = torch.topk(
                        next_token_logits, min(top_k, self.vocab_size - 1)
                    )
                    next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits_filtered.scatter_(-1, top_k_indices, top_k_values)
                    next_token_logits = next_token_logits_filtered

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_layers': len(self.blocks),
            'num_heads': self.blocks[0].attn.num_heads if self.blocks else 0,
            'ff_dim': 512,
            'max_seq_len': self.max_seq_len,
        }


def create_model(vocab_size: int = 256, **kwargs) -> TransformerForGeneration:
    """Factory function for creating transformer model"""
    return TransformerForGeneration(vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    # Test model
    model = create_model()
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch = torch.randint(0, 256, (2, 32))
    logits = model(batch)
    print(f"Forward pass: input {batch.shape} → logits {logits.shape}")

    # Test generation
    prompt = torch.tensor([[0, 1, 3]])  # [START, central_mass, num_planets]
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generation: prompt {prompt.shape} → generated {generated.shape}")
