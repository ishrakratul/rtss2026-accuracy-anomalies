# Data

## `all_6_models_inference_results_RTSS.csv`

10,000 rows, one per Tiny ImageNet test image. This is the paper's real
calibration sample -- used to derive each classifier's confidence
threshold and the PrCond/PrIDK tables the cascade algorithms consume.

| Column | Meaning |
|---|---|
| `image` | Test image filename |
| `true_label` | Ground-truth class index (0-199) |
| `<Model>_confidence` | Top-1 softmax confidence score |
| `<Model>_correct` | Whether the top-1 prediction was correct (bool) |
| `<Model>_inference_time` | Wall-clock inference time, in seconds |

`<Model>` is one of: `AlexNet`, `EfficientNet`, `VGG16`, `ResNet18`,
`ConvNeXt`, `Swin_Transformer`.

Produced by `src/inference/run_inference.py`, or regenerate it with
`run.py --mode inference` (needs trained checkpoints + a test-set labels
CSV).

## `test_labels.csv`

Only needed to regenerate the CSV above from scratch. Two columns:
`image_path` (path to a JPEG) and `tiny_label` (class index matching
Tiny-ImageNet-200's `wnids.txt` ordering). Not needed for
`run.py --mode algorithms`.
