# PIXREL dense prediction adapter

`PixrelSegmenter` connects a pretrained or randomly initialized `PixrelEncoder` to a small convolutional decoder. The decoder consumes `EncoderOutput.spatial`, so segmentation loss trains the fused multiscale pathway that the image-classification head does not use. Logits are resized to the input resolution.

## Dataset layout

Use paired image and mask directories. Relative paths must match; image extensions may differ from mask extensions. Masks must be single-channel integer class IDs, with IDs `0` through `num_classes - 1`. The default `ignore_index` is `255`.

```text
dataset/
├── train/
│   ├── images/scene-01.jpg
│   └── masks/scene-01.png
├── val/
│   ├── images/scene-02.jpg
│   └── masks/scene-02.png
└── test/
    ├── images/scene-03.jpg
    └── masks/scene-03.png
```

Palette (`P`), grayscale (`L`), and integer (`I`/`I;16`) mask images are accepted. RGB color-coded masks are rejected so color-to-class mapping cannot happen silently. Images are resized bilinearly; masks use nearest-neighbor interpolation. Training uses paired horizontal flips.

## Train and evaluate

Install the package, then pass the pretrained bundle and separate train/validation trees:

```bash
pixrel-segment --checkpoint ./encoder_bundle.pt \
  --train-images ./dataset/train/images --train-masks ./dataset/train/masks \
  --val-images ./dataset/val/images --val-masks ./dataset/val/masks \
  --num-classes 3 --class-names background object boundary \
  --epochs 30 --batch-size 8 --device auto --output ./outputs/segmentation

pixrel-segment-evaluate --checkpoint ./outputs/segmentation/best.pt \
  --images-dir ./dataset/test/images --masks-dir ./dataset/test/masks \
  --device auto --output ./outputs/segmentation/test.json
```

The trainer chooses `best.pt` by validation mean IoU. Evaluation reports mean IoU, pixel accuracy, per-class IoU, confusion matrix, valid-pixel count, and loss. Use a scratch run with the same architecture, data, and settings as the control:

```bash
pixrel-segment --scratch --variant adaptive --image-size 128 \
  --train-images ./dataset/train/images --train-masks ./dataset/train/masks \
  --val-images ./dataset/val/images --val-masks ./dataset/val/masks \
  --num-classes 3 --seed 17 --epochs 30 --batch-size 8 --device auto \
  --output ./outputs/segmentation-scratch
```

## Evidence status

The adapter has synthetic CPU tests for paired-mask loading, output resolution, gradients through the fused encoder branch, confusion-matrix metrics, and a training epoch. It has not been run on real segmentation annotations. The fire/non-fire image dataset used in the current transfer report has only image-level labels; it cannot serve as a segmentation benchmark. Real-data mIoU and scratch-versus-pretrained results are still required before claiming dense transfer or deployment readiness.

## Colab setup

After committing and pushing this code to GitHub, update the existing Colab checkout:

```bash
%%bash
set -e
cd /content/Pixrel
git pull
pip install -e .
pixrel-segment --help
```

Do not point `--train-masks` at the fire/non-fire class folders: those folders contain no pixel-level annotations. Prepare or select a separate segmentation dataset with legally usable masks, and preserve a held-out test split before training.
