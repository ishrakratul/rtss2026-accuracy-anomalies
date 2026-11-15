#!/usr/bin/env python3
"""
run.py -- single entry point for the RTSS 2026 artifact.

Modes
-----
  train              Train one or more of the six classifiers from scratch
                      (auto-downloads Tiny-ImageNet-200 if missing).
  build_test_labels  Build the evaluation set --test_labels_csv points at,
                      directly from Tiny-ImageNet-200's own official val/
                      split (val/images/ + val/val_annotations.txt) --
                      no external dataset, no account/token needed.
  inference          Run the six trained classifiers over the test set
                      and produce the calibration CSV. Auto-builds
                      --test_labels_csv first (via build_test_labels)
                      if it doesn't exist yet.
  algorithms         Reproduce Tables IV-VII and Figures 2, 3 from the
                      calibration CSV (works out of the box with the CSV
                      shipped in data/, no GPU or training required).
  all                train -> build_test_labels -> inference -> algorithms,
                      end to end.

Examples
--------
  # Fastest path: reproduce every table/figure from the shipped CSV.
  python run.py --mode algorithms --target all

  # Just Table V/VI.
  python run.py --mode algorithms --target table5_6

  # Train one classifier (Tiny-ImageNet-200 auto-downloads if missing).
  python run.py --mode train --model resnet18 --tiny_root ./tiny-imagenet-200

  # Build the evaluation set from Tiny-ImageNet-200's own val/ split.
  python run.py --mode build_test_labels --tiny_root ./tiny-imagenet-200

  # Full pipeline (requires GPU; dataset auto-downloads).
  python run.py --mode all
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

TRAIN_SCRIPTS = {
    "alexnet": "src/training/train_alexnet.py",
    "efficientnet": "src/training/train_efficientnet.py",
    "vgg16": "src/training/train_vgg16.py",
    "resnet18": "src/training/train_resnet18.py",
    "convnext": "src/training/train_convnext.py",
    "swin": "src/training/train_swin.py",
}

TABLE_FIG_SCRIPTS = {
    "table4": "src/figures/table4.py",
    "table5_6": "src/figures/table5_6.py",
    "table7": "src/figures/table7.py",
    "fig2": "src/figures/fig2_execution_time.py",
    "fig3": "src/figures/fig3_runtime_complexity.py",
}

# Scripts that don't take --data (self-contained / synthetic).
NO_DATA_TARGETS = {"fig3"}
# Scripts that don't take --outdir under outputs/tables (figures instead).
FIGURE_TARGETS = {"fig2", "fig3"}


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def do_train(args):
    models = args.model if args.model else list(TRAIN_SCRIPTS.keys())
    for m in models:
        script = TRAIN_SCRIPTS[m]
        cmd = [sys.executable, os.path.join(ROOT, script),
               "--tiny_root", args.tiny_root,
               "--save_dir", os.path.join(ROOT, "models", "pretrained"),
               "--epochs", str(args.epochs),
               "--batch_size", str(args.batch_size)]
        if args.pretrained:
            cmd.append("--pretrained")
        if args.amp:
            cmd.append("--amp")
        if args.no_download:
            cmd.append("--no_download")
        run(cmd)


def do_build_test_labels(args):
    cmd = [sys.executable, os.path.join(ROOT, "src/data_prep/build_test_labels.py"),
           "--tiny_root", args.tiny_root,
           "--out_csv", args.test_labels_csv]
    if args.no_download:
        cmd.append("--no_download")
    run(cmd)


def do_inference(args):
    if not os.path.exists(args.test_labels_csv):
        print(f"[inference] '{args.test_labels_csv}' not found -- building it from "
              f"Tiny-ImageNet-200's val/ split first...")
        do_build_test_labels(args)

    cmd = [sys.executable, os.path.join(ROOT, "src/inference/run_inference.py"),
           "--checkpoints_dir", os.path.join(ROOT, "models", "pretrained"),
           "--test_labels_csv", args.test_labels_csv,
           "--out_csv", os.path.join(ROOT, "data", "all_6_models_inference_results_RTSS.csv")]
    run(cmd)


def do_algorithms(args):
    targets = args.target if args.target != ["all"] else list(TABLE_FIG_SCRIPTS.keys())
    data_csv = os.path.join(ROOT, "data", "all_6_models_inference_results_RTSS.csv")
    for t in targets:
        script = TABLE_FIG_SCRIPTS[t]
        cmd = [sys.executable, os.path.join(ROOT, script)]
        outdir = os.path.join(ROOT, "outputs", "figures" if t in FIGURE_TARGETS else "tables")
        cmd += ["--outdir", outdir]
        if t not in NO_DATA_TARGETS:
            cmd += ["--data", data_csv]
        if t == "fig3" and args.quick:
            cmd.append("--quick")
        run(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                     choices=["train", "build_test_labels", "inference", "algorithms", "all"])

    # train options
    ap.add_argument("--model", nargs="+", choices=list(TRAIN_SCRIPTS.keys()),
                     help="Which classifier(s) to train. Default: all six.")
    ap.add_argument("--tiny_root", default="./tiny-imagenet-200",
                     help="Path to extracted tiny-imagenet-200 folder (train mode).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64,
                     help="Training batch size (train mode). Default 64 needs "
                          "~16GB GPU memory (e.g. RTX A4000); lower it (e.g. 16 or 32) "
                          "on smaller GPUs to avoid CUDA out-of-memory errors.")
    ap.add_argument("--no_download", action="store_true",
                     help="Disable auto-download of Tiny-ImageNet-200 if --tiny_root is "
                          "missing/invalid (train and build_test_labels modes).")
    ap.add_argument("--pretrained", action="store_true", help="Initialize from ImageNet weights.")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision.")

    # inference options
    ap.add_argument("--test_labels_csv", default="./data/test_labels.csv",
                     help="CSV with columns image_path,tiny_label (inference and "
                          "build_test_labels modes -- build_test_labels writes here).")

    # algorithms options
    ap.add_argument("--target", nargs="+", default=["all"],
                     choices=list(TABLE_FIG_SCRIPTS.keys()) + ["all"],
                     help="Which table(s)/figure(s) to generate. Default: all.")
    ap.add_argument("--quick", action="store_true",
                     help="Use reduced settings for fig3 (faster smoke test).")

    args = ap.parse_args()

    if args.mode == "train":
        do_train(args)
    elif args.mode == "build_test_labels":
        do_build_test_labels(args)
    elif args.mode == "inference":
        do_inference(args)
    elif args.mode == "algorithms":
        do_algorithms(args)
    elif args.mode == "all":
        do_train(args)
        do_inference(args)
        do_algorithms(args)


if __name__ == "__main__":
    main()
