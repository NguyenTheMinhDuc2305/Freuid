import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def CBR(in_ch, out_ch, k = 3, s = 1, p = 1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias = False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace = True)
    )


class PPM(nn.Module):
    def __init__(self, in_ch, out_ch, bins = (1,2,3,6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(b),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for b in bins
            ]
        )

        self.bottleneck = CBR(in_ch + len(bins) * out_ch, out_ch, k=3, p=1)
    
    def forward(self, x):
        h, w = x.shape[2:]
        feats = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        return self.bottleneck(torch.cat(feats, dim=1))

class UPerNetDecoder(nn.Module):
    """UPerNet = PPM (trên E4) + FPN (lateral trên E1..E3) -> fuse feature."""
 
    def __init__(self, enc_channels, fpn_dim=256):
        super().__init__()
        # enc_channels = [C1, C2, C3, C4] (thấp -> cao)
        self.ppm = PPM(enc_channels[-1], fpn_dim)
        self.laterals = nn.ModuleList(
            [CBR(c, fpn_dim, k=1, p=0) for c in enc_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [CBR(fpn_dim, fpn_dim, k=3, p=1) for _ in enc_channels[:-1]]
        )
        self.fuse = CBR(len(enc_channels) * fpn_dim, fpn_dim, k=3, p=1)
 
    def forward(self, feats):
        # feats = [E1, E2, E3, E4]
        ppm_out = self.ppm(feats[-1])                     # mức cao nhất của FPN
        laterals = [l(feats[i]) for i, l in enumerate(self.laterals)]
        laterals.append(ppm_out)                          # [L1, L2, L3, PPM]
 
        # top-down pathway: cộng dồn từ cao xuống thấp
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + up
 
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(self.fpn_convs))]
        outs.append(laterals[-1])                         # mức PPM giữ nguyên
 
        # upsample tất cả về kích thước mức cao nhất (E1) rồi concat + fuse
        target = outs[0].shape[2:]
        outs = [
            o if o.shape[2:] == target
            else F.interpolate(o, size=target, mode="bilinear", align_corners=False)
            for o in outs
        ]
        return self.fuse(torch.cat(outs, dim=1))
 

class EdgeAwareModule(nn.Module):
    """EAM: f^e = CBR(Cat(DR(E2), Up(DR(E4)))) -> 1 channel (edge logits).
 
    E2 (low-level, độ phân giải cao) cho biên sắc; E4 (high-level) định vị
    vùng tampered. DR = dimension reduction bằng CBR 1x1.
    """
 
    def __init__(self, c2, c4, mid=128):
        super().__init__()
        self.dr2 = CBR(c2, mid, k=1, p=0)
        self.dr4 = CBR(c4, mid, k=1, p=0)
        self.fuse = CBR(2 * mid, mid, k=3, p=1)
        self.head = nn.Conv2d(mid, 1, 1)
 
    def forward(self, e2, e4):
        x2 = self.dr2(e2)
        x4 = F.interpolate(
            self.dr4(e4), size=e2.shape[2:], mode="bilinear", align_corners=False
        )
        f = self.fuse(torch.cat([x2, x4], dim=1))
        return self.head(f)
    
"""Main class module"""
class CueNet(nn.Module):
    def __init__(
        self,
        backbone="convnextv2_tiny",
        pretrained=True,
        fpn_dim=256,
        eam_mid=128,
    ):
        super().__init__()
        # encoder: lấy 4 stage feature (stride 4/8/16/32)
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )
        ch = self.encoder.feature_info.channels()         # [C1, C2, C3, C4]
        self.enc_channels = ch
 
        # decoder + localization head
        self.decoder = UPerNetDecoder(ch, fpn_dim=fpn_dim)
        self.seg_head = nn.Conv2d(fpn_dim, 1, 1)
 
        # edge head (dùng E2 và E4)
        self.eam = EdgeAwareModule(ch[1], ch[3], mid=eam_mid)
 
        # detection head trên E4 (theo ConvNeXt v2: pool -> norm -> linear)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(ch[3]),
            nn.Linear(ch[3], 1),
        )
 
    def forward(self, x):
        H, W = x.shape[2:]
        E1, E2, E3, E4 = self.encoder(x)
 
        seg = self.seg_head(self.decoder([E1, E2, E3, E4]))
        seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
 
        edge = self.eam(E2, E4)
        edge = F.interpolate(edge, size=(H, W), mode="bilinear", align_corners=False)
 
        cls = self.cls_head(E4)
        return {"seg": seg, "cls": cls, "edge": edge}     # logits
 
    @torch.no_grad()
    def predict(self, x):
        out = self.forward(x)
        return {
            "mask": torch.sigmoid(out["seg"]),
            "score": torch.sigmoid(out["cls"]),
            "edge": torch.sigmoid(out["edge"]),
        }
 