"""Multi-task forgery model for Stage-A pretraining: detection (binary) + localization
(mask). EdgeDoc/TruFor-style — a timm encoder + lightweight FPN mask head + a GAP
classification head. The shared forensic features then transfer to Stage-B (FREUID).
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskForgery(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True,
                 dec_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = timm.create_model(backbone, features_only=True,
                                         pretrained=pretrained)
        chs = self.encoder.feature_info.channels()           # e.g. [96,192,384,768]
        cfg = self.encoder.pretrained_cfg
        self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])
        self.laterals = nn.ModuleList([nn.Conv2d(c, dec_dim, 1) for c in chs])
        self.smooth = nn.Conv2d(dec_dim, dec_dim, 3, padding=1)
        self.mask_head = nn.Conv2d(dec_dim, 1, 1)
        self.cls_head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(chs[-1], 256), nn.GELU(),
                                      nn.Linear(256, 1))

    def trainable_parameters(self):                          # full fine-tune
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        feats = self.encoder(x)                              # list, fine->coarse
        p = self.laterals[-1](feats[-1])                     # FPN top-down
        for i in range(len(feats) - 2, -1, -1):
            p = F.interpolate(p, size=feats[i].shape[-2:], mode="nearest") \
                + self.laterals[i](feats[i])
        mask = self.mask_head(self.smooth(p))                # (B,1,H/4,W/4) logits
        cls = self.cls_head(feats[-1].mean(dim=(2, 3))).squeeze(-1)   # (B,) logit
        return cls, mask


def dice_bce_mask_loss(mask_logit, mask_gt):
    """BCE + soft Dice on the localization mask."""
    mg = F.interpolate(mask_gt, size=mask_logit.shape[-2:], mode="area")
    mg = (mg > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(mask_logit, mg)
    p = torch.sigmoid(mask_logit)
    dice = 1 - (2 * (p * mg).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + mg.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


if __name__ == "__main__":
    m = MultiTaskForgery()
    print("encoder channels:", m.encoder.feature_info.channels(),
          "| mean:", m.mean)
    x = torch.randn(2, 3, 384, 384)
    cls, mask = m(x)
    print("cls:", cls.shape, "| mask:", mask.shape)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"params: {n:.1f}M")
    gt = (torch.rand(2, 1, 384, 384) > 0.9).float()
    print("mask loss:", float(dice_bce_mask_loss(mask, gt)))
