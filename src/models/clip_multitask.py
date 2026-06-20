"""CLIP-backbone multi-task model = best of both:
  - CLIP ViT-L foundation features (generalize well — gave public 0.0566), FROZEN
    except LayerNorm (LN-tuning), like the CLIP-linear baseline.
  - EdgeDoc-style two heads: detection (CLS token) + localization (patch grid -> mask).

Trained semi-weakly (src/train_semiweak.py): cls loss on all, mask loss only where a
mask exists. forward(x) -> (cls_logit, mask_logit), same interface as MultiTaskForgery.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT = "vit_large_patch14_clip_224.openai"


class CLIPMultiTask(nn.Module):
    def __init__(self, backbone: str = DEFAULT, img_size: int | None = None,
                 pretrained: bool = True, tune_norm: bool = True,
                 l2_norm: bool = True, dropout: float = 0.0, dec_dim: int = 256,
                 unfreeze_blocks: int = 0):
        super().__init__()
        kw = {"img_size": img_size} if img_size else {}
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, **kw)
        cfg = self.backbone.pretrained_cfg
        self.input_size = img_size or cfg["input_size"][-1]
        self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])
        self.npref = self.backbone.num_prefix_tokens
        self.l2_norm = l2_norm
        C = self.backbone.embed_dim

        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(C, 1))
        self.mask_dec = nn.Sequential(                       # patch grid -> mask (×4)
            nn.Conv2d(C, dec_dim, 1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dec_dim, dec_dim, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dec_dim, 1, 1))

        for p in self.backbone.parameters():                 # LN-tuning
            p.requires_grad = False
        if tune_norm:
            for n, p in self.backbone.named_parameters():
                if "norm" in n:
                    p.requires_grad = True
        # unfreeze the last K transformer blocks so the MASK gradient can reshape
        # the top spatial features (real multi-task coupling, not just LN)
        if unfreeze_blocks > 0 and hasattr(self.backbone, "blocks"):
            for blk in self.backbone.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        feats = self.backbone.forward_features(x)            # (B, npref+N, C)
        cls = feats[:, 0]
        patches = feats[:, self.npref:]                      # (B, N, C)
        B, N, C = patches.shape
        g = int(N ** 0.5)
        grid = patches.transpose(1, 2).reshape(B, C, g, g)   # (B, C, g, g)
        if self.l2_norm:
            cls = F.normalize(cls, dim=-1)
        return self.cls_head(cls).squeeze(-1), self.mask_dec(grid)


if __name__ == "__main__":
    m = CLIPMultiTask(img_size=224)
    tr = sum(p.numel() for p in m.trainable_parameters()) / 1e6
    tot = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"trainable {tr:.2f}M / total {tot:.1f}M | mean={m.mean} input={m.input_size}")
    x = torch.randn(2, 3, 224, 224)
    cls, mask = m(x)
    print("cls:", tuple(cls.shape), "| mask:", tuple(mask.shape))
