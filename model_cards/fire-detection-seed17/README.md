---
library_name: pixrel
tags:
  - computer-vision
  - image-encoder
  - self-supervised-learning
  - pixel-relationships
---

# PIXREL adaptive encoder — fire image transfer pilot

Hugging Face repository: [`nexusbert/pixrel-fire-encoder-v1`](https://huggingface.co/nexusbert/pixrel-fire-encoder-v1).

> Research prototype. This checkpoint is not validated for fire alarms, emergency response, or other safety-critical use.

This repository contains the PIXREL adaptive encoder pretrained on the training split of a small fire/non-fire image classification dataset. The encoder was trained with masked teacher-feature prediction and masked RGB reconstruction. The Hub files include `encoder_bundle.pt`, which can be loaded and fine-tuned with the PIXREL Python package.

## Status and evidence

This is an early, task-specific transfer experiment, not a general-purpose vision foundation model. On this experiment's held-out test split (152 images), the scratch classifier scored higher than the pretrained classifier for all three reported seeds. The results do not support a claim that this pretraining improves fire-image classification.

The consolidated project report and linked experiment records are available in the [PIXREL GitHub repository](https://github.com/neuralnex/Pixrel/blob/main/docs/final_report.md).

| Seed | Pretrained accuracy | Scratch accuracy | Pretrained loss | Scratch loss |
|---:|---:|---:|---:|---:|
| 17 | 94.74% | 94.74% | 0.2940 | 0.1853 |
| 29 | 93.42% | 96.05% | 0.3558 | 0.1858 |
| 43 | 94.08% | 96.71% | 0.2953 | 0.2047 |
| Mean across seeds | **94.08%** | **95.83%** | **0.3150** | **0.1919** |

The mean pretrained accuracy is 1.75 percentage points below the scratch baseline. The test split contains 152 images; estimates from this small set have substantial uncertainty. The reported test values were supplied from Colab evaluation runs. This model card does not include per-class metrics or confidence intervals.

## Intended use and limitations

- Research and experimentation with pixel-level image representations and transfer learning.
- Fine-tuning on image classification tasks with an ImageFolder-style labeled dataset.
- Not validated on external fire datasets, different cameras, geographic regions, weather, smoke-only scenes, or real-time video.
- Not a fire alarm or safety system. Do not use it to make emergency decisions.
- The training dataset source, version, and redistribution terms have not been established here. Users must verify the original dataset terms before redistributing weights or data.
- The model is small (approximately 150k encoder parameters) and trained at 128×128 resolution for 20 epochs on an NVIDIA L4. Results should not be generalized beyond the reported setup.

## Load and fine-tune

Install PIXREL and the Hub client in a Python 3.10+ environment:

```bash
pip install "pixrel[hub]"
```

Download the bundle and fine-tune it with labeled class folders:

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="OWNER/REPOSITORY",
    filename="encoder_bundle.pt",
)
print(checkpoint)
```

```bash
pixrel-transfer --checkpoint /path/from/previous/cell/encoder_bundle.pt \
  --train-dir /path/to/task/train --val-dir /path/to/task/val \
  --epochs 30 --batch-size 32 --device auto --output ./outputs/task-pretrained
```

Each split must contain one directory per class. Keep the test split separate and evaluate it only after selecting settings on validation data. See the [PIXREL repository](https://github.com/neuralnex/Pixrel) for the training code and benchmark protocol.

## Training and visualization artifacts

The associated pretraining run used the fire dataset's training split only, seed 17, adaptive variant, 128×128 images, 20 epochs, patch size 4, batch size 8, and RGB reconstruction weight 0.1. Selected learning curves, feature visualizations, and `pretrain.json` are provided when included in this Hub repository. Feature-map visualizations are normalized diagnostics; they are not causal explanations.

## License

No model license is declared yet because the source dataset's terms and the intended model redistribution rights have not been verified. Check those rights and select a license before inviting downstream redistribution.
