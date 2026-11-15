"""
build_test_labels.py

Builds the test set that src/inference/run_inference.py expects
(--test_labels_csv), directly from Tiny-ImageNet-200's own OFFICIAL
val/ split -- no external dataset, no Hugging Face account/token.
Auto-downloads Tiny-ImageNet-200 to --tiny_root if it's missing (same
auto-download used by src/training/train_*.py); pass --no_download to
disable that and fail with a clear error instead.

Tiny-ImageNet-200 ships val/images/val_0.JPEG ... val_9999.JPEG (10,000
images, 50 per class) plus val/val_annotations.txt, a tab-separated file
mapping each filename to its class WNID:

    val_0.JPEG    n03444034    0  32  44  62
    val_1.JPEG    n04067472    52 55  57  59
    ...

This script reads that file, maps each WNID to its Tiny-ImageNet class
index via wnids.txt (the same index the trained classifiers' output
heads use), and writes a labels.csv with (image_path, tiny_label) --
exactly the format run_inference.py --test_labels_csv expects. This is
the actual source of the paper's evaluation set (confirmed by the
shipped calibration CSV's own filenames: val_0.JPEG, val_1.JPEG, ...).

Usage:
    python build_test_labels.py \
        --tiny_root ../../tiny-imagenet-200 \
        --out_csv ../../data/test_labels.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.utils import ensure_tiny_imagenet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny_root", default="../../tiny-imagenet-200",
                     help="Path to Tiny-ImageNet-200 (needs val/images/ and val/val_annotations.txt). "
                          "Auto-downloaded here if missing/invalid.")
    ap.add_argument("--out_csv", default="../../data/test_labels.csv",
                     help="Where to write the (image_path, tiny_label) CSV.")
    ap.add_argument("--no_download", action="store_true",
                     help="Disable auto-download of Tiny-ImageNet-200 if --tiny_root is missing/invalid.")
    args = ap.parse_args()

    ensure_tiny_imagenet(args.tiny_root, auto_download=not args.no_download)

    wnids_path = os.path.join(args.tiny_root, "wnids.txt")
    val_images_dir = os.path.join(args.tiny_root, "val", "images")
    val_annotations_path = os.path.join(args.tiny_root, "val", "val_annotations.txt")

    for p in (wnids_path, val_images_dir, val_annotations_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' not found. Tiny-ImageNet-200 at '{args.tiny_root}' doesn't have the "
                f"expected val/ layout (wnids.txt, val/images/, val/val_annotations.txt) even "
                f"after the download/validity check -- the folder may be corrupted or from a "
                f"non-standard source. Try removing --tiny_root and re-running to re-download."
            )

    # WNID -> Tiny-ImageNet-200 class index (same ordering the trained
    # classifiers' 200-way output heads use).
    with open(wnids_path) as f:
        wnids = [line.strip() for line in f if line.strip()]
    wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}
    print(f"Tiny-ImageNet-200 classes: {len(wnids)}")

    # val_annotations.txt: filename \t wnid \t x0 \t y0 \t x1 \t y1  (no header)
    annotations = pd.read_csv(
        val_annotations_path, sep="\t", header=None,
        names=["filename", "wnid", "x0", "y0", "x1", "y1"],
    )
    print(f"val_annotations.txt rows: {len(annotations)}")

    rows = []
    missing = 0
    for _, row in annotations.iterrows():
        wnid = row["wnid"]
        if wnid not in wnid_to_label:
            missing += 1
            continue
        image_path = os.path.join(val_images_dir, row["filename"])
        rows.append({"image_path": image_path, "tiny_label": wnid_to_label[wnid]})

    if missing:
        print(f"[Warn] {missing} val_annotations.txt rows referenced a WNID not in wnids.txt (skipped).")

    labels_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    labels_df.to_csv(args.out_csv, index=False)

    print(f"\nWrote {len(labels_df)} rows to {args.out_csv}")
    print(f"Classes covered: {labels_df['tiny_label'].nunique()} / {len(wnids)}")
    print(f"\nNext step:")
    print(f"  python run.py --mode inference --test_labels_csv {args.out_csv}")


if __name__ == "__main__":
    main()
