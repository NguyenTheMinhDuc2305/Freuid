"""CLIP ViT-L + LayerNorm-tuning + linear head — classification-only forgery detector.

Recipe (LNCLIP-DF, "Deepfake Detection that Generalizes Across Benchmarks", 2025):
  - frozen CLIP ViT-L/14 backbone (generalizable foundation features)
  - unfreeze ONLY LayerNorm params (~0.03%) -> parameter-efficient, anti-overfit
  - L2-normalize the CLS feature, then a single linear head -> 1 logit (binary)

Only image-level labels are used (no masks). See docs/baseline_proposal.md §0b.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BACKBONE = "vit_large_patch14_clip_224.openai"


class CLIPLinearForgery(nn.Module):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 tune_norm: bool = True, l2_norm: bool = True, dropout: float = 0.0,
                 img_size: int | None = None):
        super().__init__()
        # img_size lets us run the card at higher resolution (keeps fine forgery
        # traces). ViT needs img_size at build time to interpolate pos-embeds;
        # ConvNeXt/CNN backbones are resolution-agnostic and reject the kwarg.
        kw = {}
        if img_size is not None:
            try:
                timm.create_model(backbone, pretrained=False, num_classes=0,
                                  img_size=img_size)
                kw["img_size"] = img_size
            except TypeError:
                pass  # backbone ignores img_size (e.g. ConvNeXt) — feed any size
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, **kw)
        cfg = self.backbone.pretrained_cfg
        self.input_size = img_size or cfg["input_size"][-1]
        self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])
        feat_dim = self.backbone.num_features
        self.l2_norm = l2_norm
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, 1)

        # freeze everything, then re-enable LayerNorm (+ the head is always trainable)
        for p in self.backbone.parameters():
            p.requires_grad = False
        if tune_norm:
            for name, p in self.backbone.named_parameters():
                if "norm" in name:        # LayerNorm weight/bias
                    p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        feat = self.backbone(x)                 # (B, feat_dim) CLS token
        if self.l2_norm:
            feat = F.normalize(feat, dim=-1)
        return self.head(self.drop(feat)).squeeze(-1)   # (B,) logit


def count_params(model: nn.Module):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tot


if __name__ == "__main__":
    m = CLIPLinearForgery()
    tr, tot = count_params(m)
    print(f"backbone input={m.input_size} mean={m.mean}")
    print(f"trainable {tr/1e6:.3f}M / total {tot/1e6:.1f}M ({100*tr/tot:.2f}%)")
    x = torch.randn(2, 3, m.input_size, m.input_size)
    print("logits:", m(x).shape)
