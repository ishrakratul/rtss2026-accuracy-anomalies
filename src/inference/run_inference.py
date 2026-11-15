"""
run_inference.py

Loads the six trained classifiers (AlexNet, EfficientNet-B0, VGG16,
ResNet18, ConvNeXt-Tiny, Swin-Base) and runs each over the Tiny ImageNet
test set, recording per-sample confidence, correctness, and inference time.
Writes the combined CSV that src/cascade and src/figures consume
(matches the format of data/all_6_models_inference_results_RTSS.csv).

Expects checkpoints in models/pretrained/ with these exact filenames
(produced by the scripts in src/training/):
    best_model_alexnet_paper.pth
    best_efficientnet_b0_model_paper.pth
    best_model_vgg16_paper.pth
    best_resnet18_model_paper.pth
    convnext_tiny_tinyimagenet_best.pt
    best_model_swin.pth

Expects a test set as a list of (image_path, true_label) -- by default,
built from a directory of images plus a labels.csv with columns
(image_path, tiny_label). See --test_labels_csv.

Usage:
    python run_inference.py \
        --checkpoints_dir ../../models/pretrained \
        --test_labels_csv ../../data/test_labels.csv \
        --out_csv ../../data/all_6_models_inference_results_RTSS.csv
"""
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
import timm

WARMUP_RUNS = 50

imagenet_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def clean_state_dict(ckpt):
    """Supports raw state_dict, {"state_dict": ...}, {"model_state_dict": ...},
    {"model": ...}. Also strips common "module."/"model." prefixes."""
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    cleaned = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    return cleaned


def load_model_weights(model, state_dict_path, device, strict=True):
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Model state dictionary not found at '{state_dict_path}'")
    ckpt = torch.load(state_dict_path, map_location=device)
    state_dict = clean_state_dict(ckpt)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        print(f"RuntimeError loading {state_dict_path} with strict=True:")
        print(e)
        print("Trying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_alexnet(device, ckpt_dir):
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, 200)
    return load_model_weights(model, os.path.join(ckpt_dir, "best_model_alexnet_paper.pth"), device)


def build_efficientnet(device, ckpt_dir):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 200)
    return load_model_weights(model, os.path.join(ckpt_dir, "best_efficientnet_b0_model_paper.pth"), device)


def build_vgg16(device, ckpt_dir):
    model = models.vgg16(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 200)
    return load_model_weights(model, os.path.join(ckpt_dir, "best_model_vgg16_paper.pth"), device)


def build_resnet18(device, ckpt_dir):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    return load_model_weights(model, os.path.join(ckpt_dir, "best_resnet18_model_paper.pth"), device)


def build_swin_transformer(device, ckpt_dir):
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=200)
    return load_model_weights(model, os.path.join(ckpt_dir, "best_model_swin.pth"), device)


def build_convnext(device, ckpt_dir):
    model = torchvision.models.convnext_tiny(weights=None)
    for i in reversed(range(len(model.classifier))):
        if isinstance(model.classifier[i], nn.Linear):
            in_feats = model.classifier[i].in_features
            model.classifier[i] = nn.Linear(in_feats, 200)
            break
    return load_model_weights(model, os.path.join(ckpt_dir, "convnext_tiny_tinyimagenet_best.pt"), device)


def warmup_model(model, model_col_name, test_images, preprocess, device, warmup_runs):
    print(f"{model_col_name}: Running {warmup_runs} warmup runs...")
    model.eval()
    with torch.no_grad():
        for i in range(warmup_runs):
            image_path, _ = test_images[i % len(test_images)]
            try:
                image_orig = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Warmup error loading image {image_path}: {e}")
                continue
            image = preprocess(image_orig).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
    print(f"{model_col_name}: Warmup completed.")


def run_inference_for_model(model, model_col_name, test_images, preprocess, device, results_df):
    total_test, correct_test, total_inference_time = 0, 0, 0.0
    confidences, correct_flags, inference_times = [], [], []

    print(f"\nRunning inference for {model_col_name}...")
    warmup_model(model, model_col_name, test_images, preprocess, device, WARMUP_RUNS)

    for idx, (image_path, true_label) in enumerate(test_images):
        if idx % 1000 == 0 and idx != 0:
            print(f"{model_col_name}: Processing image {idx}/{len(test_images)}...")
        try:
            image_orig = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            confidences.append(np.nan)
            correct_flags.append(False)
            inference_times.append(np.nan)
            continue

        image = preprocess(image_orig).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            confidence_value = confidence.item()
            predicted_label = predicted.item()
            inference_time = time.time() - start_time

        total_inference_time += inference_time
        total_test += 1
        is_correct = predicted_label == true_label
        if is_correct:
            correct_test += 1

        confidences.append(confidence_value)
        correct_flags.append(is_correct)
        inference_times.append(inference_time)

    accuracy = correct_test / total_test if total_test > 0 else 0
    avg_time = total_inference_time / total_test if total_test > 0 else 0
    print(f"{model_col_name} Total Decision Time: {total_inference_time:.4f} seconds")
    print(f"{model_col_name} Average Decision Time per Image: {avg_time:.6f} seconds")
    print(f"{model_col_name} Test Accuracy: {accuracy:.4f}")

    results_df[f"{model_col_name}_confidence"] = confidences
    results_df[f"{model_col_name}_correct"] = correct_flags
    results_df[f"{model_col_name}_inference_time"] = inference_times
    return results_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", default="../../models/pretrained")
    ap.add_argument("--test_labels_csv", default="../../data/test_labels.csv",
                     help="CSV with columns: image_path, tiny_label")
    ap.add_argument("--out_csv", default="../../data/all_6_models_inference_results_RTSS.csv")
    ap.add_argument("--models", nargs="+",
                     default=["AlexNet", "EfficientNet", "VGG16", "ResNet18", "ConvNeXt", "Swin_Transformer"],
                     help="Subset of models to run inference for.")
    args = ap.parse_args()

    if not os.path.exists(args.test_labels_csv):
        raise FileNotFoundError(
            f"--test_labels_csv '{args.test_labels_csv}' not found.\n\n"
            f"Build it from Tiny-ImageNet-200's own val/ split (no external "
            f"dataset needed) with:\n\n"
            f"  python src/data_prep/build_test_labels.py \\\n"
            f"      --tiny_root ./tiny-imagenet-200 \\\n"
            f"      --out_csv {args.test_labels_csv}\n\n"
            f"(or: python run.py --mode build_test_labels --tiny_root ./tiny-imagenet-200)\n"
            f"Then re-run this with --test_labels_csv pointing at that file."
        )

    labels_df = pd.read_csv(args.test_labels_csv)
    test_images = [(row["image_path"], int(row["tiny_label"])) for _, row in labels_df.iterrows()]
    print(f"Loaded {len(test_images)} test images from {args.test_labels_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_df = pd.DataFrame({
        "image": [os.path.basename(p) for p, _ in test_images],
        "true_label": [y for _, y in test_images],
    })

    builders = {
        "AlexNet": build_alexnet,
        "EfficientNet": build_efficientnet,
        "VGG16": build_vgg16,
        "ResNet18": build_resnet18,
        "ConvNeXt": build_convnext,
        "Swin_Transformer": build_swin_transformer,
    }

    for name in args.models:
        if name not in builders:
            raise ValueError(f"Unknown model {name}; choices are {list(builders)}")
        model = builders[name](device, args.checkpoints_dir)
        results_df = run_inference_for_model(model, name, test_images, imagenet_preprocess, device, results_df)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved combined results to: {args.out_csv}")
    print(results_df.head())


if __name__ == "__main__":
    main()
