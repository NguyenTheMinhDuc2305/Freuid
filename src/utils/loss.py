import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(1)
    union = prob.sum(1) + target.sum(1)
    return (1 - (2 * inter + eps) / (union + eps)).mean()

def weighted_bce(logits, target, eps=1e-6, max_w=100.0):
    """BCE cân bằng lớp theo từng ảnh: pixel giả (hiếm) được trọng số cao hơn."""
    b = target.shape[0]
    flat = target.flatten(1)
    pos = flat.sum(1, keepdim=True)                        # số pixel tampered
    total = flat.shape[1]
    pos_w = ((total - pos) / (pos + eps)).clamp(1.0, max_w).view(b, 1, 1, 1)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(target > 0.5, pos_w, torch.ones_like(pos_w))
    return (loss * weight).mean()


class CueNetLoss(nn.Module):
    """L_hard = α·L_seg + β·(L_cls + L_edg)
 
    L_seg = λ·wbce + (1-λ)·dice ;  L_cls = bce ;  L_edg = dice
    Mặc định theo paper: α=1, β=0.2, λ=0.1.
    """
 
    def __init__(self, alpha=1.0, beta=0.2, lam_seg=0.1):
        super().__init__()
        self.alpha, self.beta, self.lam = alpha, beta, lam_seg
 
    def forward(self, out, labels):
        l_seg = self.lam * weighted_bce(out["seg"], labels["seg"]) + (
            1 - self.lam
        ) * dice_loss(out["seg"], labels["seg"])
        l_cls = F.binary_cross_entropy_with_logits(out["cls"], labels["cls"])
        l_edg = dice_loss(out["edge"], labels["edge"])
        l_hard = self.alpha * l_seg + self.beta * (l_cls + l_edg)
        return l_hard, {
            "seg": l_seg.item(), "cls": l_cls.item(),
            "edge": l_edg.item(), "hard": l_hard.item(),
        }