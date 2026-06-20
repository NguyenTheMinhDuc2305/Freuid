"""Forensic (noise-residual) forgery detector — baseline ĐA DẠNG cho ensemble.

Khác 3 baseline semantic (CLIP/DINO/ConvNeXt ăn RGB): model này ăn **NOISE RESIDUAL**
trích bằng 3 filter SRM cố định (high-pass). Residual bỏ phần "nội dung nhìn thấy" (template,
chân dung) → model KHÔNG học thuộc template được (chống overfit), chỉ thấy **bất nhất nhiễu**
ở đường nối face-swap/inpaint. → tương quan THẤP với các model RGB → đẩy ensemble + tốt cho
private (xem docs/reports/backbone-and-dataset-analysis.md, UAM-Biometrics dùng TruFor thắng private).

Tái dùng CLIPLinearForgery: chỉ chèn SRM trước backbone. SRM cố định (buffer) → không lưu thêm,
vẫn chỉ train LayerNorm + linear head. Backbone nào cũng dùng được (--backbone).
"""
import torch
import torch.nn.functional as F

from src.models.clip_classifier import CLIPLinearForgery

# 3 filter SRM kinh điển (Fridrich-Kodovsky; dùng trong RGB-N / MVSS-Net), 5x5.
_SRM = [
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
]
_SRM_NORM = [4.0, 12.0, 2.0]   # hệ số chuẩn hoá từng filter


def _srm_weight():
    """(out=3, in=3, 5, 5): mỗi filter SRM cộng qua 3 kênh RGB -> 1 kênh noise. Tổng 3 kênh out."""
    w = torch.zeros(3, 3, 5, 5)
    for k in range(3):
        kern = torch.tensor(_SRM[k], dtype=torch.float32) / _SRM_NORM[k]
        for c in range(3):
            w[k, c] = kern
    return w


class ForensicForgery(CLIPLinearForgery):
    """SRM noise-residual -> backbone (frozen + LN-tune) -> linear head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("srm", _srm_weight())   # cố định, không trainable, không cần lưu

    def forward(self, x):
        x = F.conv2d(x, self.srm, padding=2)          # RGB(normalized) -> 3ch noise residual
        return super().forward(x)                     # backbone -> l2norm -> head


if __name__ == "__main__":
    from src.models.clip_classifier import count_params
    m = ForensicForgery(backbone="convnextv2_large.fcmae_ft_in22k_in1k",
                        pretrained=False, img_size=512)
    tr, tot = count_params(m)
    x = torch.randn(2, 3, 512, 512)
    print(f"forensic out={tuple(m(x).shape)} trainable={tr/1e6:.3f}M/{tot/1e6:.0f}M "
          f"srm_fixed={tuple(m.srm.shape)} (requires_grad={m.srm.requires_grad})")
