# Pretrained checkpoints

Put the six trained checkpoints here with these **exact** filenames
(the inference/calibration code looks them up by name):

| Filename | Architecture |
|---|---|
| `best_model_alexnet_paper.pth` | AlexNet |
| `best_efficientnet_b0_model_paper.pth` | EfficientNet-B0 |
| `best_model_vgg16_paper.pth` | VGG16 |
| `best_resnet18_model_paper.pth` | ResNet18 |
| `convnext_tiny_tinyimagenet_best.pt` | ConvNeXt-Tiny |
| `best_model_swin.pth` | Swin-Base (swin_base_patch4_window7_224) |

Each is produced by the matching script in `src/training/` -- e.g.
`train_resnet18.py --save_dir models/pretrained` writes
`best_resnet18_model_paper.pth` here directly.

Already have the checkpoints? Just copy them in -- `run.py --mode
inference` and `run.py --mode all` pick them up automatically.

Format: each file is a dict with keys `model_name`, `img_size`,
`num_classes`, `state_dict`.
