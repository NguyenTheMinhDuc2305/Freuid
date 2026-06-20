"""P1 (print-and-capture simulation) + P3 (resolution/quality jitter) augmentation.

albumentations 2.x. Forensic-preserving: NO horizontal flip (text direction is a
real cue) and no global filter strong enough to wipe local forgery traces.
See docs/reports/train-vs-test-stats.md for why these specific augs.
"""
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def _amplitude_jitter(image, **kwargs):
    """FACT-style style randomization: perturb the low-frequency AMPLITUDE spectrum
    (global illumination/style) while keeping PHASE (structure). Single-image, so
    it simulates capture/style shift — relevant to the captured private test."""
    img = image.astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    amp, pha = np.abs(f), np.angle(f)
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    r = max(1, int(min(h, w) * 0.08))                      # low-freq window
    scale = np.random.uniform(0.7, 1.3)
    amp[cy - r:cy + r, cx - r:cx + r] *= scale
    out = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha), axes=(0, 1)), axes=(0, 1)).real
    return np.clip(out, 0, 255).astype(np.uint8)


def _fourier_block(p=0.5):
    return A.Lambda(name="amplitude_jitter", image=_amplitude_jitter, p=p)


def build_p1p3(img_size, mean, std, fourier=False):
    return A.Compose([
        A.Resize(img_size, img_size),                         # standardize first (fast)
        # --- P1: geometric (card photographed at an angle / tilted) ---
        A.Rotate(limit=5, border_mode=0, fill=0, p=0.4),
        A.Perspective(scale=(0.02, 0.06), keep_size=True, border_mode=0, fill=0, p=0.3),
        # --- P3: resolution / quality jitter ---
        A.OneOf([
            A.Downscale(scale_range=(0.5, 0.9), p=1.0),       # low-res capture
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),         # mild defocus
        ], p=0.4),
        A.ImageCompression(quality_range=(40, 95), p=0.6),    # phone re-save
        # --- P1: lighting / print-capture ---
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.OneOf([
            A.RandomShadow(shadow_intensity_range=(0.2, 0.5), p=1.0),
            A.RandomSunFlare(src_radius=80, p=1.0),           # glare / specular
        ], p=0.2),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
        A.ISONoise(p=0.2),                                    # sensor / paper grain
        *([_fourier_block(p=0.5)] if fourier else []),        # style randomization
        # --- finalize ---
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_fourier_only(img_size, mean, std):
    """Light base (resize) + Fourier amplitude jitter — isolate the style-aug effect."""
    return A.Compose([
        A.Resize(img_size, img_size),
        _fourier_block(p=0.7),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_aug(name, img_size, mean, std):
    """Dispatch an augmentation preset (returns an albumentations Compose).
    name in: p1p3 | fourier | p1p3_fourier."""
    if name == "p1p3":
        return build_p1p3(img_size, mean, std, fourier=False)
    if name == "p1p3_fourier":
        return build_p1p3(img_size, mean, std, fourier=True)
    if name == "fourier":
        return build_fourier_only(img_size, mean, std)
    raise ValueError(f"unknown aug preset: {name}")


class AlbuWrapper:
    """Adapt an albumentations Compose to the PIL-in / tensor-out interface
    expected by FraudIDDataset.transform."""

    def __init__(self, atransform):
        self.t = atransform

    def __call__(self, pil_img):
        return self.t(image=np.asarray(pil_img))["image"]


if __name__ == "__main__":
    from PIL import Image
    IM_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IM_STD = (0.26862954, 0.26130258, 0.27577711)
    tf = AlbuWrapper(build_p1p3(224, IM_MEAN, IM_STD))
    img = Image.open("DATA/train/0000e24edb864abe8b0defa703742dcb.jpeg").convert("RGB")
    out = tf(img)
    print("ok, output:", out.shape, out.dtype, "range",
          round(float(out.min()), 2), round(float(out.max()), 2))
