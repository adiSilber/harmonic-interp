#!/usr/bin/env python3
"""
Download the Aria model weights the reproduce pipeline needs.

Fetches `model-gen.safetensors` (~2.6 GB) from the public HuggingFace repo
`loubb/aria-medium-base` into `config/models/aria-medium-base/` — the path the
pipeline's checkpoint (`reproduce/consts.py: CHECKPOINT`) expects. With this, a fresh
clone can run end-to-end without hand-placing the weights.

The tiny model-config JSONs (`config/models/medium-*.json`) are already in the repo;
only the large weight file is downloaded here.

Usage:
    python reproduce/download_model.py
    python reproduce/download_model.py --force   # re-download even if present
"""

import argparse
import os

from huggingface_hub import hf_hub_download

REPO_ID = "loubb/aria-medium-base"
FILENAME = "model-gen.safetensors"
DEST_DIR = "config/models/aria-medium-base"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = p.parse_args()

    dest = os.path.join(DEST_DIR, FILENAME)
    if os.path.exists(dest) and not args.force:
        print(f"Already present ({os.path.getsize(dest) / 1e9:.1f} GB): {dest}")
        print("(pass --force to re-download)")
        return

    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading {FILENAME} (~2.6 GB) from {REPO_ID} -> {DEST_DIR}/ ...")
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=DEST_DIR)
    print(f"Done. Weights at: {path}")


if __name__ == "__main__":
    main()
