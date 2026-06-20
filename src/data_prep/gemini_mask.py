"""Đánh MASK vùng giả mạo cho ảnh train FREUID bằng Gemini API -> weak localization label.

FREUID train chỉ có nhãn ảnh (0/1), KHÔNG có mask. Utility này nhờ Gemini segment vùng
trông bị **chỉnh sửa/dán** (chân dung swap, text inpaint) -> xuất PNG mask + manifest, để
bật nhánh localization (multi-task kiểu đội vô địch DeepID). Đây là **weak label** (Gemini
không phải forgery-detector chuẩn) nhưng đủ làm tín hiệu phụ + tăng data.

Cài + key:
    pip install google-genai
    export GEMINI_API_KEY=...        # https://aistudio.google.com/apikey

Dùng (LUÔN thử --limit nhỏ trước để soát chất lượng + chi phí):
    python -m src.data_prep.gemini_mask --limit 30 --only-fraud 1
    python -m src.data_prep.gemini_mask --sample 2000 --concurrency 4     # mẻ lớn (resume được)

Output:
    DATA/gemini_masks/<id>.png         (mask nhị phân 0/255, đúng HxW ảnh gốc)
    DATA/gemini_masks/manifest.csv     (id,image_path,label,mask_ref="png:<abs>")  -> nạp vào loader
Mask dùng được ngay nhờ handler "png:" trong src/dataset/combined.py:_render_mask.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "DATA")

PROMPT_FORGERY = (
    "You are a forensic analyst for identity documents. This ID image is KNOWN to be "
    "digitally manipulated. Find the region(s) that were edited/pasted/inpainted — typically "
    "the PORTRAIT photo (face-swap, with blending seams or a pasted rectangle) and/or TEXT "
    "fields (mismatched font, alignment, spacing, or compression). "
    "Output ONLY a JSON list; each item has 'box_2d' ([ymin,xmin,ymax,xmax] normalized 0-1000), "
    "'mask' (base64 PNG probability map sized to the box), and 'label'. "
    "Prefer the single most suspicious region; return [] only if truly nothing looks altered."
)
PROMPT_REGIONS = (
    "Segment the editable regions of this identity document: the PORTRAIT photo and each "
    "filled TEXT field (name, dates, numbers). Output a JSON list; each item has 'box_2d' "
    "([ymin,xmin,ymax,xmax] normalized 0-1000), 'mask' (base64 PNG sized to the box), 'label'."
)


def _client(api_key):
    try:
        from google import genai  # noqa: PLC0415
    except ImportError:
        sys.exit("Thiếu SDK. Chạy: pip install google-genai  (và export GEMINI_API_KEY=...)")
    return genai.Client(api_key=api_key)


def _parse_json(text):
    """Bóc JSON list từ text Gemini (có thể bọc ```json ... ```)."""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S) or re.search(r"(\[.*\])", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return []


def _compose_mask(items, H, W):
    """Ghép các box+mask base64 (toạ độ 0-1000) -> mask nhị phân HxW."""
    out = np.zeros((H, W), np.uint8)
    for it in items:
        box = it.get("box_2d")
        if not box or len(box) != 4:
            continue
        y0, x0, y1, x1 = box
        y0, y1 = sorted((int(y0 / 1000 * H), int(y1 / 1000 * H)))
        x0, x1 = sorted((int(x0 / 1000 * W), int(x1 / 1000 * W)))
        y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(H, y1), min(W, x1)
        if y1 <= y0 or x1 <= x0:
            continue
        raw = it.get("mask", "")
        if isinstance(raw, str) and "base64," in raw:
            raw = raw.split("base64,", 1)[1]
        try:
            mimg = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L").resize((x1 - x0, y1 - y0))
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], (np.asarray(mimg) > 127).astype(np.uint8))
        except Exception:
            out[y0:y1, x0:x1] = 1            # mask hỏng -> dùng nguyên box làm fallback
    return out * 255


def _annotate_one(client, model, prompt, row, out_dir, retries=3):
    dst = os.path.join(out_dir, f"{row.id}.png")
    if os.path.exists(dst):
        return row.id, "skip", f"png:{dst}"
    path = os.path.join(DATA, row.image_path)
    try:
        img = Image.open(path).convert("RGB")
    except OSError:
        return row.id, "bad_img", None
    for k in range(retries):
        try:
            r = client.models.generate_content(model=model, contents=[img, prompt])
            items = _parse_json(r.text or "")
            mask = _compose_mask(items, img.height, img.width)
            Image.fromarray(mask).save(dst)
            return row.id, ("empty" if mask.max() == 0 else "ok"), f"png:{dst}"
        except Exception as e:                # rate-limit / transient -> backoff
            if k == retries - 1:
                return row.id, f"err:{type(e).__name__}", None
            time.sleep(2 * (k + 1))


def parse_args():
    p = argparse.ArgumentParser(description="Gemini-annotated forgery masks for FREUID train")
    p.add_argument("--csv", default=os.path.join(DATA, "train_labels.csv"))
    p.add_argument("--out-dir", default=os.path.join(DATA, "gemini_masks"))
    p.add_argument("--manifest", default=os.path.join(DATA, "gemini_masks", "manifest.csv"))
    p.add_argument("--only-fraud", type=int, default=1, help="1 = chỉ ảnh giả (label==1)")
    p.add_argument("--sample", type=int, default=0, help=">0: lấy ngẫu nhiên N ảnh")
    p.add_argument("--limit", type=int, default=0, help=">0: chỉ N ảnh đầu (test nhanh)")
    p.add_argument("--prompt-mode", default="forgery", choices=["forgery", "regions"])
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Chưa có key. export GEMINI_API_KEY=...  (https://aistudio.google.com/apikey)")
    os.makedirs(args.out_dir, exist_ok=True)
    client = _client(key)
    prompt = PROMPT_FORGERY if args.prompt_mode == "forgery" else PROMPT_REGIONS

    df = pd.read_csv(args.csv)
    if args.only_fraud:
        df = df[df.label == 1]
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed)
    if args.limit:
        df = df.head(args.limit)
    print(f"sẽ đánh mask {len(df)} ảnh | model={args.model} mode={args.prompt_mode} "
          f"-> {args.out_dir} (resume: bỏ qua ảnh đã có)")

    rows, stats = [], {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(_annotate_one, client, args.model, prompt, r, args.out_dir)
                for r in df.itertuples(index=False)]
        for i, f in enumerate(as_completed(futs), 1):
            rid, status, ref = f.result()
            stats[status.split(":")[0]] = stats.get(status.split(":")[0], 0) + 1
            if ref:
                rr = df[df.id == rid].iloc[0]
                rows.append({"id": rid, "image_path": rr.image_path, "label": int(rr.label),
                             "mask_ref": ref})
            if i % 25 == 0 or i == len(df):
                print(f"  {i}/{len(df)} | {stats}")

    if rows:
        man = pd.DataFrame(rows)
        if os.path.exists(args.manifest):                      # gộp với manifest cũ (resume)
            man = pd.concat([pd.read_csv(args.manifest), man]).drop_duplicates("id", keep="last")
        man.to_csv(args.manifest, index=False)
        print(f"manifest: {len(man)} dòng -> {args.manifest}")
    print(f"XONG. stats={stats}")


if __name__ == "__main__":
    main()
