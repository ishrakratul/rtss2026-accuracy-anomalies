"""
dataset.py

Auto-download helper for the Tiny-ImageNet-200 dataset used by
src/training/train_*.py. If --tiny_root doesn't look like a valid
Tiny-ImageNet-200 folder, this downloads and extracts the official
Stanford CS231n archive (~237MB) into it automatically.
"""
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _looks_valid(tiny_root: str) -> bool:
    """A valid Tiny-ImageNet-200 folder has wnids.txt and a train/ dir
    with per-class subfolders."""
    wnids_path = os.path.join(tiny_root, "wnids.txt")
    train_dir = os.path.join(tiny_root, "train")
    if not (os.path.isfile(wnids_path) and os.path.isdir(train_dir)):
        return False
    # Cheap sanity check: at least a handful of class subfolders exist.
    try:
        return len(os.listdir(train_dir)) >= 100
    except OSError:
        return False


def _download_with_progress(url: str, dest_path: str):
    def _report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading tiny-imagenet-200.zip: {pct:3d}%  ({mb_done:6.1f} / {mb_total:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_report)
    print()  # newline after progress bar


def ensure_tiny_imagenet(tiny_root: str, auto_download: bool = True) -> str:
    """Make sure `tiny_root` points at a valid Tiny-ImageNet-200 folder,
    downloading and extracting it there if it's missing.

    Returns tiny_root unchanged (for convenient chaining). Raises
    FileNotFoundError if the folder is missing/invalid and
    auto_download=False.
    """
    if _looks_valid(tiny_root):
        return tiny_root

    if not auto_download:
        raise FileNotFoundError(
            f"'{tiny_root}' does not look like a valid Tiny-ImageNet-200 folder "
            f"(expected wnids.txt and a train/ directory with class subfolders). "
            f"Download it manually from {TINY_IMAGENET_URL} and extract it there, "
            f"or drop --no_download to auto-download."
        )

    print(f"[dataset] '{tiny_root}' not found or incomplete -- auto-downloading Tiny-ImageNet-200...")
    os.makedirs(os.path.dirname(os.path.abspath(tiny_root)) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "tiny-imagenet-200.zip")
        _download_with_progress(TINY_IMAGENET_URL, zip_path)

        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        extracted_root = os.path.join(tmp_dir, "tiny-imagenet-200")
        if not os.path.isdir(extracted_root):
            raise RuntimeError(
                f"Downloaded archive did not contain the expected 'tiny-imagenet-200/' folder."
            )

        # Move into place at the caller-requested path.
        target = os.path.abspath(tiny_root)
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.move(extracted_root, target)

    if not _looks_valid(tiny_root):
        raise RuntimeError(f"Download/extraction finished but '{tiny_root}' still isn't a valid dataset folder.")

    print(f"[dataset] Tiny-ImageNet-200 ready at '{tiny_root}'.")
    return tiny_root
