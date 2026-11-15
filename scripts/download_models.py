"""Download the six trained checkpoints into models/pretrained/.

The checkpoints (~1.3 GB in total, CC-BY-4.0) are archived on Zenodo:
https://doi.org/10.5281/zenodo.24243817

    python3 scripts/download_models.py

Files that are already present with the correct checksum are skipped.
Only needed for Option B (re-running inference); Option A works without
any checkpoints.
"""
import hashlib
import os
import sys
import urllib.request

ZENODO_RECORD = "24243817"

# filename -> MD5 checksum published on the Zenodo record
FILES = {
    "best_model_alexnet_paper.pth": "b8489ace5066daf9f1e73719168755f2",
    "best_efficientnet_b0_model_paper.pth": "6cf93733ab8a7d64cfd0d464fc9cb75c",
    "best_model_vgg16_paper.pth": "08b190c9d9a4fa5bf4928cb58df05d52",
    "best_resnet18_model_paper.pth": "b08ab6475189901bd53e1446b46c1ee3",
    "convnext_tiny_tinyimagenet_best.pt": "ac210891c3d994e53528e2f660881feb",
    "best_model_swin.pth": "5d6959cc13a46048627d3faa1d8a3e2c",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "models", "pretrained")


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(DEST, exist_ok=True)
    failed = []
    for name, expected in FILES.items():
        path = os.path.join(DEST, name)
        if os.path.exists(path) and md5(path) == expected:
            print(f"[skip] {name} (already present)")
            continue
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
        print(f"[get ] {name}")
        urllib.request.urlretrieve(url, path)
        if md5(path) != expected:
            print(f"[FAIL] {name}: checksum mismatch")
            failed.append(name)
    if failed:
        sys.exit(f"{len(failed)} file(s) failed the checksum check; re-run to retry.")
    print(f"Done. All six checkpoints are in {DEST}")


if __name__ == "__main__":
    main()
