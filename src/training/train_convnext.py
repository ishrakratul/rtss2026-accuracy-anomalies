# train_convnext.py
# Trains (single GPU) on Tiny-ImageNet-200 with ConvNeXt.
# Uses EXACTLY N images per class from TRAIN as validation (does NOT use official val split).
#
# Features:
# - optional --pretrained (ImageNet)
# - train on Tiny-ImageNet TRAIN split, evaluate on held-out subset of TRAIN (val) each epoch
# - inference check on 5 random val samples every epoch
# - saves best checkpoint (inference-ready: clean state_dict + metadata)
# - optional --freeze_backbone (train classifier head only)
# - optional --amp
# - CUDA cleanup helpers
#
# Fix included: forces images to 3 channels (RGB) to avoid Normalize broadcast errors.

import os
import gc
import argparse
import random
import sys
import time
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as T
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base, convnext_large
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.utils import ensure_tiny_imagenet

# -----------------------------
# Repro + CUDA cleanup
# -----------------------------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_cuda_mem(tag: str = ""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024**2)
        r = torch.cuda.memory_reserved() / (1024**2)
        print(f"[CUDA mem] {tag} allocated={a:.1f} MiB reserved={r:.1f} MiB")

def cleanup_cuda(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# -----------------------------
# Model helpers
# -----------------------------
def input_size_for_model(model_name: str) -> int:
    return 224

def create_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    # Robust weights handling across torchvision versions
    def _weights_for(factory_name: str):
        if not pretrained:
            return None
        weights_cls_name = {
            "convnext_tiny": "ConvNeXt_Tiny_Weights",
            "convnext_small": "ConvNeXt_Small_Weights",
            "convnext_base": "ConvNeXt_Base_Weights",
            "convnext_large": "ConvNeXt_Large_Weights",
        }[factory_name]
        weights_cls = getattr(torchvision.models, weights_cls_name, None)
        return getattr(weights_cls, "DEFAULT", None) if weights_cls is not None else "DEFAULT"

    if model_name == "convnext_tiny":
        m = convnext_tiny(weights=_weights_for("convnext_tiny"))
    elif model_name == "convnext_small":
        m = convnext_small(weights=_weights_for("convnext_small"))
    elif model_name == "convnext_base":
        m = convnext_base(weights=_weights_for("convnext_base"))
    elif model_name == "convnext_large":
        m = convnext_large(weights=_weights_for("convnext_large"))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Replace classifier head (last Linear in Sequential)
    if isinstance(m.classifier, nn.Sequential):
        replaced = False
        for i in reversed(range(len(m.classifier))):
            if isinstance(m.classifier[i], nn.Linear):
                in_feats = m.classifier[i].in_features
                m.classifier[i] = nn.Linear(in_feats, num_classes)
                replaced = True
                break
        if not replaced:
            raise RuntimeError("Could not find Linear layer in ConvNeXt classifier to replace.")
    else:
        raise RuntimeError("Unexpected ConvNeXt classifier type; expected nn.Sequential.")

    return m

def freeze_backbone_if_requested(model: nn.Module, freeze_backbone: bool):
    if not freeze_backbone:
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


# -----------------------------
# Tiny-ImageNet dataset helpers
# -----------------------------
class TinyImageNetPaths(Dataset):
    """
    Dataset from a list of (image_path, label).
    Reads with torchvision.io.read_image -> forces 3 channels -> converts to PIL -> applies transforms.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # Read image as uint8 tensor [C,H,W], but C might be 1 (grayscale), 3 (RGB), or 4 (RGBA)
        img = torchvision.io.read_image(path)

        # Force 3 channels to match ImageNet normalization
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)   # [1,H,W] -> [3,H,W]
        elif img.shape[0] == 4:
            img = img[:3, :, :]         # drop alpha

        img = img.float() / 255.0
        img = T.ToPILImage()(img)       # now RGB-like PIL image

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def load_wnids(wnids_path: str) -> List[str]:
    with open(wnids_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def collect_train_samples(train_root: str, wnids: List[str]) -> List[Tuple[str, int]]:
    # train/<wnid>/images/*.JPEG
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
    samples: List[Tuple[str, int]] = []
    for wnid in wnids:
        img_dir = os.path.join(train_root, wnid, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing directory: {img_dir}")
        label = wnid_to_label[wnid]
        for fn in os.listdir(img_dir):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                samples.append((os.path.join(img_dir, fn), label))
    return samples

def split_train_val_exact_per_class(
    samples: List[Tuple[str, int]],
    num_classes: int,
    val_per_class: int,
    seed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Exactly val_per_class images per class go to validation, rest stay in training.
    """
    rnd = random.Random(seed)
    by_class: Dict[int, List[Tuple[str, int]]] = {c: [] for c in range(num_classes)}
    for path, y in samples:
        by_class[y].append((path, y))

    train_out, val_out = [], []
    for c in range(num_classes):
        items = by_class[c]
        rnd.shuffle(items)

        if len(items) < val_per_class:
            raise ValueError(
                f"Class {c} has only {len(items)} samples, cannot take val_per_class={val_per_class}."
            )

        val_out.extend(items[:val_per_class])
        train_out.extend(items[val_per_class:])

    rnd.shuffle(train_out)
    rnd.shuffle(val_out)
    return train_out, val_out


# -----------------------------
# Transforms / Loaders
# -----------------------------
def build_transforms(img_size: int, pretrained: bool):
    # Tiny-ImageNet is ImageNet-like; ImageNet normalization is a good default.
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return train_tf, val_tf

def build_loaders_tinyimagenet(
    tiny_root: str,
    img_size: int,
    batch_size: int,
    workers: int,
    pretrained: bool,
    val_per_class: int,
    seed: int,
):
    wnids_path = os.path.join(tiny_root, "wnids.txt")
    train_root = os.path.join(tiny_root, "train")

    wnids = load_wnids(wnids_path)
    num_classes = len(wnids)
    if num_classes != 200:
        print(f"[Warn] wnids count is {num_classes}, expected 200 (continuing anyway).")

    all_train_samples = collect_train_samples(train_root, wnids)

    train_samples, val_samples = split_train_val_exact_per_class(
        all_train_samples,
        num_classes=num_classes,
        val_per_class=val_per_class,
        seed=seed
    )

    train_tf, val_tf = build_transforms(img_size, pretrained)

    train_ds = TinyImageNetPaths(train_samples, transform=train_tf)
    val_ds   = TinyImageNetPaths(val_samples, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False
    )

    class_names = wnids  # wnid strings
    return train_ds, val_ds, train_loader, val_loader, class_names, num_classes


# -----------------------------
# Train / Eval / Inference check
# -----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total

@torch.no_grad()
def inference_five_samples(model, ds, device, class_names):
    model.eval()
    idxs = random.sample(range(len(ds)), 5)
    print("  [Inference check: 5 random VAL samples]")
    for i, idx in enumerate(idxs, 1):
        x, y = ds[idx]
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).item()
        print(f"    #{i}: true={y}({class_names[y]}), pred={pred}({class_names[pred]})")


# -----------------------------
# Save (inference-ready)
# -----------------------------
def save_for_inference(path: str, model: nn.Module, model_name: str, img_size: int, num_classes: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_name": model_name,
        "img_size": img_size,
        "num_classes": num_classes,
        "state_dict": model.state_dict(),
    }, path)


# -----------------------------
# Optimizer helpers
# -----------------------------
def choose_optimizer(model, args):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if args.adamw:
        opt = torch.optim.AdamW(params, lr=args.lr_adamw, weight_decay=args.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        return opt, sch

    opt = torch.optim.SGD(params, lr=args.lr_sgd, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    return opt, sch


# -----------------------------
# Run one model
# -----------------------------
def run_model(model_name: str, args, device):
    print("\n" + "=" * 90)
    print(f"Running model: {model_name}")
    print("=" * 90)

    img_size = input_size_for_model(model_name)

    train_ds, val_ds, train_loader, val_loader, class_names, num_classes = build_loaders_tinyimagenet(
        tiny_root=args.tiny_root,
        img_size=img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        pretrained=args.pretrained,
        val_per_class=args.val_per_class,
        seed=args.seed,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Classes: {num_classes}")
    print(f"Val per class: {args.val_per_class} (total val = {args.val_per_class * num_classes})")

    model = create_model(model_name, num_classes=num_classes, pretrained=args.pretrained).to(device)
    freeze_backbone_if_requested(model, args.freeze_backbone)

    optimizer, scheduler = choose_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_acc = 0.0
    start = time.time()
    save_path = os.path.join(args.save_dir, f"{model_name}_tinyimagenet_best.pt")

    print_cuda_mem(f"{model_name} start")

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp)
        va_loss, va_acc = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = (time.time() - start) / 60.0

        print(
            f"\nEpoch {epoch+1:03d}/{args.epochs} | bs={args.batch_size} | lr={lr_now:.6f} | elapsed={elapsed:.1f}m\n"
            f"  Train: loss={tr_loss:.4f}, acc={tr_acc:.4f}\n"
            f"  Val  : loss={va_loss:.4f}, acc={va_acc:.4f}"
        )

        inference_five_samples(model, val_ds, device, class_names)

        if va_acc > best_acc:
            best_acc = va_acc
            save_for_inference(save_path, model, model_name, img_size, num_classes)
            print(f"  [Saved BEST] {save_path} (best val acc={best_acc:.4f})")

    print(f"\n{model_name} best val accuracy: {best_acc:.4f}")

    print_cuda_mem(f"{model_name} before cleanup")
    cleanup_cuda(model, optimizer, scheduler, scaler, train_loader, val_loader, train_ds, val_ds, class_names)
    print_cuda_mem(f"{model_name} after cleanup")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tiny_root", type=str, default="./tiny-imagenet-200",
                        help="Path to extracted tiny-imagenet-200 folder")
    parser.add_argument("--no_download", action="store_true",
                        help="Disable auto-download of Tiny-ImageNet-200 if --tiny_root is missing/invalid")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--model", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--val_per_class", type=int, default=50,
                        help="Exact number of TRAIN images per class to use for validation")

    # Optimizer knobs
    parser.add_argument("--lr_sgd", type=float, default=0.01)
    parser.add_argument("--lr_adamw", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--adamw", action="store_true", help="Use AdamW instead of SGD")
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers_prefetch", type=int, default=2,
                        help="(Unused) placeholder in case you extend DataLoader prefetching")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print_cuda_mem("startup")

    # Tiny-ImageNet train has 500 images/class
    if args.val_per_class < 1:
        raise ValueError("--val_per_class must be >= 1")
    if args.val_per_class >= 500:
        print("[Warn] Tiny-ImageNet has 500 train images/class. "
              "If val_per_class is 500, training set becomes empty.")

    ensure_tiny_imagenet(args.tiny_root, auto_download=not args.no_download)

    run_model(args.model, args, device)
    cleanup_cuda()
    print_cuda_mem("done")

    print("\nAll done.")

if __name__ == "__main__":
    main()

