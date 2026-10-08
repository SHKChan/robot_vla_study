import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768, depth=12, heads=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # (224/16)^2 = 196

        # A conv whose kernel size equals its stride computes exactly the
        # "cut into patches, flatten, apply one shared linear layer" operation
        # -- in a single fused op, with the output already laid out as a grid.
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # A learnable token prepended to the sequence; after the transformer
        # its output vector serves as the whole-image summary.
        # std 0.02 matters: torch.randn's unit std is ~50x too large.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Attention is permutation-invariant, so position must be injected
        # explicitly. +1 accounts for the CLS token.
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)

        # norm_first=True selects pre-LayerNorm. The PyTorch default is
        # post-LN, which needs careful warmup to train at depth 12 and
        # otherwise diverges. Every modern ViT and LLM uses pre-LN.
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch_embed(x)                          # [B, 3, 224, 224] -> [B, D, 14, 14]
        # Flatten the 14x14 grid into a length-196 sequence, then move the
        # channel axis last so the shape is [batch, tokens, features].
        x = x.flatten(2).transpose(1, 2)                 # [B, 196, D]

        # expand (not repeat) broadcasts the single CLS token across the batch
        # without copying memory.
        cls = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                   # [B, 197, D]

        x = x + self.pos_embed                           # broadcast over batch
        x = self.transformer(x)                          # [B, 197, D]
        return self.norm(x)[:, 0]                        # take CLS -> [B, D]


if __name__ == '__main__':
    model = ViT()
    out = model(torch.randn(2, 3, 224, 224))
    print(out.shape)  # torch.Size([2, 768])
