"""Download the full IDNet-2025 (HuggingFace cactuslab/IDNet-2025) into DATA/IDNet.

Per-country tar.gz are downloaded, extracted to DATA/IDNet/extracted/<COUNTRY>/,
then the tar is deleted to keep disk usage ~ extracted size only. Safe to re-run:
already-extracted countries are skipped.
"""
import os
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "DATA", "IDNet")
EXTRACT = os.path.join(DATA, "extracted")
os.makedirs(EXTRACT, exist_ok=True)


def main():
    api = HfApi()
    info = api.repo_info("cactuslab/IDNet-2025", repo_type="dataset", files_metadata=True)
    tars = sorted([s.rfilename for s in info.siblings if s.rfilename.endswith(".tar.gz")])
    print(f"{len(tars)} country archives to fetch", flush=True)

    for i, fn in enumerate(tars, 1):
        country = fn.replace(".tar.gz", "")
        done_marker = os.path.join(EXTRACT, f".{country}.done")
        if os.path.exists(done_marker):
            print(f"[{i}/{len(tars)}] {country}: already done, skip", flush=True)
            continue
        # remove any leftover partial tar so hf re-downloads cleanly
        stale = os.path.join(DATA, fn)
        if os.path.exists(stale):
            os.remove(stale)
        ok = False
        for attempt in (1, 2):                 # retry once on corrupt download
            print(f"[{i}/{len(tars)}] {country}: downloading (try {attempt}) ...", flush=True)
            path = hf_hub_download("cactuslab/IDNet-2025", fn, repo_type="dataset",
                                   local_dir=DATA, force_download=(attempt == 2))
            print(f"[{i}/{len(tars)}] {country}: extracting ...", flush=True)
            # NOTE: -xf (auto-detect), NOT -xzf: some IDNet files are named .tar.gz
            # but are PLAIN (uncompressed) tar -> forcing gzip (-z) would fail.
            if subprocess.run(["tar", "-xf", path, "-C", EXTRACT]).returncode == 0:
                ok = True
                break
            print(f"[{i}/{len(tars)}] {country}: tar corrupt, re-downloading", flush=True)
            os.remove(path)
        if not ok:
            print(f"[{i}/{len(tars)}] {country}: FAILED twice, skipping", flush=True)
            continue
        os.remove(path)                        # free the tar.gz immediately
        open(done_marker, "w").close()
        print(f"[{i}/{len(tars)}] {country}: done (tar removed)", flush=True)

    print("ALL IDNET DOWNLOADED", flush=True)


if __name__ == "__main__":
    main()
