"""Download the six trained checkpoints into models/pretrained/.

The checkpoints (~1.3 GB in total) are too large for GitHub, so they are
hosted on Zenodo. Set ZENODO_RECORD below to the record number once the
upload is published, then run:

    python3 scripts/download_models.py

Files that already exist are skipped. Only needed for Option B
(re-running inference); Option A works without any checkpoints.
"""
import os
import sys
import urllib.request

ZENODO_RECORD = "24243817"  # <-- replace with the Zenodo record number

FILES = [
    "best_model_alexnet_paper.pth",
    "best_efficientnet_b0_model_paper.pth",
    "best_model_vgg16_paper.pth",
    "best_resnet18_model_paper.pth",
    "convnext_tiny_tinyimagenet_best.pt",
    "best_model_swin.pth",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "models", "pretrained")


def main():
    if not ZENODO_RECORD.isdigit():
        sys.exit("Set ZENODO_RECORD in scripts/download_models.py first.")
    os.makedirs(DEST, exist_ok=True)
    for name in FILES:
        path = os.path.join(DEST, name)
        if os.path.exists(path):
            print(f"[skip] {name} already present")
            continue
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
        print(f"[get ] {name}")
        urllib.request.urlretrieve(url, path)
    print(f"Done. Checkpoints are in {DEST}")


if __name__ == "__main__":
    main()
