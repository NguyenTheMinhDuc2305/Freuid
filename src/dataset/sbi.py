"""Self-Blended Images (SBI) for ID cards — data-centric generalization (§1.2).

Turns a REAL card into a SYNTHETIC forgery by blending a mildly-transformed copy
of the portrait region back with a feathered mask. The model then learns the
GENERIC blending-boundary artifact (colour/illumination/edge mismatch) instead of
a specific generator's fingerprint -> generalizes to unseen manipulations.

Used on label-0 (real) images; the produced image is labelled 1 (fake).
"""
import cv2
import numpy as np

# portrait box (fractions of W,H) — same region used across the EDA
PORTRAIT_BOX = (0.02, 0.30, 0.27, 0.80)


def _source_transform(region):
    """Mild appearance + geometric change on the copy that gets blended back."""
    r = region.astype(np.float32)
    # colour / brightness shift
    r = r * np.random.uniform(0.85, 1.15, 3) + np.random.uniform(-12, 12, 3)
    if np.random.rand() < 0.5:                       # slight blur (resample mismatch)
        k = np.random.choice([3, 5])
        r = cv2.GaussianBlur(r, (k, k), 0)
    h, w = region.shape[:2]
    if np.random.rand() < 0.7:                       # small affine -> geometric seam
        dx, dy = np.random.uniform(-0.03, 0.03, 2) * [w, h]
        sc = np.random.uniform(0.97, 1.03)
        M = np.float32([[sc, 0, dx], [0, sc, dy]])
        r = cv2.warpAffine(r, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(r, 0, 255)


def _feather_mask(h, w):
    """Soft elliptical mask covering most of the portrait, blurred at the edge."""
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * np.random.uniform(0.32, 0.46)),
                 int(h * np.random.uniform(0.32, 0.46))),
                0, 0, 360, 1.0, -1)
    sigma = np.random.uniform(0.04, 0.10) * min(h, w)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)


def self_blend(img: np.ndarray, box=PORTRAIT_BOX) -> np.ndarray:
    """img: uint8 HWC RGB -> uint8 HWC RGB with a blended portrait (synthetic fake)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))
    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return img
    src = _source_transform(region)
    m = _feather_mask(region.shape[0], region.shape[1])[..., None]
    blended = region.astype(np.float32) * (1 - m) + src * m
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    img = np.asarray(Image.open(
        "DATA/train/000514f0340642f3b2eb83bff862458a.jpeg").convert("RGB"))  # a REAL one
    fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    ax[0, 0].imshow(img); ax[0, 0].set_title("REAL (original)", fontsize=9)
    for i in range(1, 8):
        r, c = divmod(i, 4)
        ax[r, c].imshow(self_blend(img)); ax[r, c].set_title(f"SBI fake {i}", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig("out/sbi_preview.png", dpi=85, bbox_inches="tight")
    print("saved out/sbi_preview.png")
