import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Three separate projections of the SAME input.
        # Separating key from value lets the model match on one representation
        # and retrieve a different one.
        self.query: nn.Linear = nn.Linear(dim, dim)
        self.key: nn.Linear = nn.Linear(dim, dim)
        self.value: nn.Linear = nn.Linear(dim, dim)

        # Output projection: mixes information back into the residual stream.
        # W_O: mixes back into the residual stream
        self.out: nn.Linear = nn.Linear(dim, dim)

        # 1/sqrt(d_k). Without it, dot products grow as sqrt(d_k),
        # the softmax saturates toward one-hot, and gradients vanish.
        # NOTE: for multi-head this must be the PER-HEAD key dim, not dim.
        # 1/sqrt(d_k), and d_k = dim for one head
        self.scale = dim **-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # [B, N, D]
        # x: [B, N, D]-- B sequences of N tokens, each D-dimensional
        Q, K, V = self.query(x), self.key(x), self.value(x) # [B, N, D]

        # Q @ K^T pairs every token with every other -> [B, N, N] scores.
        # transpose(-2,-1) swaps the last two axes, leaving batch alone.
        scores: torch.Tensor = Q @ K.transpose(-2,-1) * self.scale # [B, N, N]
        # softmax over the LAST axis: each row sums to 1, giving the weights
        # token i assigns to every token j.
        attn: torch.Tensor = torch.softmax(scores, dim=-1) # [B, N, N]
        # Weighted sum of values-> [B, N, D]
        return self.out(attn @ V)

# In practice use the built-in: it handles multiple heads, masking and
# dropout, and dispatches to a fused (FlashAttention) kernel.
# attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
# out, weights = attn(x, x, x)  # self-attention: query = key = value

# Multi-head attention splits $D$ evenly into $h$ heads of size $d_k = D/h$:
# def out(x)->torch.Tensor:
#     pass

# def mha(x, h)->torch.Tensor:                                         # x: [B, N, D]
#     B, N, D = x.shape
#     d = D // h
#     Q, K, V = (proj(x).view(B, N, h, d).transpose(1, 2) for proj in (q, k, v))  # [B, h, N, d]
#     A = torch.softmax(Q @ K.transpose(-2, -1) / d**0.5, dim=-1)                 # [B, h, N, N]
#     return out((A @ V).transpose(1, 2).reshape(B, N, D))                        # concat → [B, N, D]