# PIXREL

PIXREL (Pixel Relationship Learning) is an experimental computer vision research project. This repository starts with a reproducible implementation of a pixel relationship encoder and the controls needed to test whether adaptive local relationships add value over convolution or fixed neighborhood aggregation. Training can run on CPU or CUDA GPUs.

No superiority or foundation-model claims are made. Reported metrics are limited to the specific datasets, protocols, and runs documented in the experiment reports.

The consolidated current-stage outcome is in the [PIXREL final research report](docs/final_report.md), including the CIFAR-100 results, fire-image pretraining diagnostics, held-out transfer comparison, and release status.

## Quick start

Use Python 3.10+ and install the package and dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Run a CIFAR-100 experiment (downloads the dataset on first run). `--device auto` uses CUDA when PyTorch can see a GPU, otherwise CPU:

```bash
pixrel-train --variant adaptive --seed 17 --epochs 20 --data ./data --output ./outputs/adaptive-seed17 --device auto
pixrel-evaluate --checkpoint ./outputs/adaptive-seed17/best.pt --data ./data --output ./outputs/adaptive-seed17/eval.json --device auto
```

Variants are `adaptive`, `fixed`, and `conv`. Run each with the same seeds and training settings. Begin with `--epochs 1 --limit-train-batches 10 --limit-val-batches 5` as a pipeline smoke run. These limited runs are not benchmark results. Use `--device cpu` to force CPU, `--device cuda` to require a visible CUDA GPU, or `--device auto` to select automatically. Use `--threads` to control CPU parallelism.

The CIFAR download progress reaching `100%` only marks the download/extraction step; training follows afterward. The script prints when data is ready, when each epoch begins, and a status line every 10 training batches.

## Research protocol

The first benchmark compares the three variants on CIFAR-100 using identical data splits, augmentation, optimizer, schedule, epochs, and predetermined seeds (17, 29, 43). The training set is split into train/validation; the official test set is evaluated only after model selection. Report mean and standard deviation across seeds, parameter counts, latency, and the full run configuration. A short run validates engineering only. Do not tune on the test set.

The controls share a hierarchical encoder skeleton. `adaptive` learns content-dependent neighbor weights, `fixed` uses uniform valid-neighbor weights, and `conv` uses a local convolution block. These are initial controls, not guaranteed parameter/FLOP-matched baselines. Record parameter counts and computational costs before interpreting differences.

See [docs/architecture.md](docs/architecture.md), [docs/benchmark_protocol.md](docs/benchmark_protocol.md), and [docs/research_blueprint.md](docs/research_blueprint.md).

The first three-seed CIFAR-100 results and their limitations are documented in [docs/experiments/relationship-value-v1.md](docs/experiments/relationship-value-v1.md). They are exploratory classification results, not evidence that PIXREL is a foundation model.

## Self-supervised learning pilot

The initial pretraining command uses unlabeled images from the 45,000-image CIFAR-100 training subset. It predicts clean teacher features from masked student inputs at H0, at the fused spatial output, and at the global H3 embedding. RGB heads at H0–H3 also predict masked input pixels; the run reports masked RGB loss, MAE, and PSNR. CIFAR-100 is only a pipeline and representation-learning pilot, not a foundation-model corpus.

```bash
pixrel-pretrain --variant adaptive --seed 17 --epochs 100 --batch-size 128 --device cuda --data ./data --output ./outputs/pretrain-cifar100
```

The encoder weights are written to `encoder.pt`, and `encoder_bundle.pt` adds variant, input resolution, and normalization metadata. `last.pt` includes teacher, predictors, optimizer, scheduler, and RNG state for resume. After pretraining, run a frozen linear probe:

```bash
pixrel-train --variant adaptive --seed 17 --epochs 20 --batch-size 64 --device cuda --init-encoder ./outputs/pretrain-cifar100/encoder.pt --freeze-encoder --data ./data --output ./outputs/probe-cifar100-seed17
```

For a random-frozen-feature control, use the same command without `--init-encoder` and save to a separate output directory. For end-to-end fine-tuning, pass `--init-encoder` and omit `--freeze-encoder`.

### ImageFolder pretraining and transfer

For broader unlabeled image collections, point `pixrel-pretrain --dataset imagefolder` at any existing directory containing supported images. Images can sit directly in that directory or in nested folders; labels and folder names are ignored during pretraining. The output includes `encoder_bundle.pt`, which stores encoder weights, variant, input size, and normalization. CIFAR-100 remains the default dataset for reproducing the pilot above.

Example GPU run in Colab (start with a small batch; larger image sizes use more memory):

```bash
pixrel-pretrain --dataset imagefolder --variant adaptive --seed 17 --epochs 100 --image-size 128 --patch-size 4 --batch-size 16 --device cuda --data /content/unlabeled-images --output /content/drive/MyDrive/pixrel-runs/pixrel-imagefolder-128
```

Pretraining automatically saves `visualizations/epoch_*.png`, `feature_path_progress.gif`, and `learning_curves.png`. The snapshots show the fixed input and mask, H0–H3 RGB reconstructions and masked-pixel error, teacher/student feature responses, each encoder stage, fused spatial maps, and learned 3×3 neighbor weights at the center pixel. The curves include masked-pixel MAE and PSNR. Feature maps are normalized separately to reveal spatial patterns; their colors are not calibrated activation magnitudes, and local weights are diagnostics rather than causal explanations.

Transfer the resulting encoder to a labeled ImageFolder classification task. Use either `--freeze-encoder` for a linear probe or omit it to fine-tune the encoder. The data root must contain one subdirectory per class; pass a separate validation root with `--val-dir`, or the CLI creates a seeded validation split.

```bash
pixrel-transfer --checkpoint /content/drive/MyDrive/pixrel-runs/pixrel-imagefolder-128/encoder_bundle.pt --train-dir /content/task/train --val-dir /content/task/val --epochs 30 --batch-size 32 --device cuda --output /content/drive/MyDrive/pixrel-runs/task-finetune
pixrel-transfer-evaluate --checkpoint /content/drive/MyDrive/pixrel-runs/task-finetune/best.pt --data-dir /content/task/test --device cuda --output /content/drive/MyDrive/pixrel-runs/task-finetune/test.json
```

For dense prediction, `pixrel-segment` trains a segmentation decoder on the encoder's fused spatial representation. Images and single-channel class-ID masks live in parallel trees with matching relative names (for example `train/images/case1.jpg` and `train/masks/case1.png`). Use `255` for ignored pixels. It selects checkpoints by validation mean IoU and reports per-class IoU and a confusion matrix. The adapter passes synthetic CPU tests, but it has not been evaluated on real masks; the fire/non-fire dataset has image-level labels only:

```bash
pixrel-segment --checkpoint /path/to/encoder_bundle.pt \
  --train-images /data/train/images --train-masks /data/train/masks \
  --val-images /data/val/images --val-masks /data/val/masks \
  --num-classes 3 --class-names background object boundary \
  --epochs 30 --batch-size 8 --device auto --output ./outputs/segmentation
pixrel-segment-evaluate --checkpoint ./outputs/segmentation/best.pt \
  --images-dir /data/test/images --masks-dir /data/test/masks \
  --device auto --output ./outputs/segmentation/test.json
```

This dense-task adapter has synthetic CPU tests, but it has not yet been evaluated on a real mask-labeled dataset. The existing fire/non-fire image dataset has class labels, not pixel masks, and cannot validate segmentation. See [the segmentation guide](docs/segmentation.md) and [final report](docs/final_report.md).

To create reviewed binary flame masks for a 200-image pilot before training, follow the [Colab fire-mask annotation guide](docs/colab_fire_mask_annotation.md). It preserves the existing split, generates background masks for non-fire images, and interactively outlines visible flame regions in fire images.

Choose image collections whose terms permit this use, and record their source, version, filtering, and split. The new ImageFolder and transfer paths are implementation scaffolding; validate them with a short run before committing GPU time to a long experiment.

See [docs/foundation_model_plan.md](docs/foundation_model_plan.md) before interpreting this pilot or scaling to a larger image corpus.

### Hugging Face Hub and community fine-tuning

The first fire-image transfer comparison is documented in [docs/experiments/fire-transfer-v1.md](docs/experiments/fire-transfer-v1.md). On this small test split, the scratch baseline averaged 95.83% accuracy and pretrained PIXREL averaged 94.08% across three seeds; this checkpoint is shared as an experimental starting point, not a demonstrated improvement. The model card is at [model_cards/fire-detection-seed17/README.md](model_cards/fire-detection-seed17/README.md).

Install the optional Hub client with `pip install -e '.[hub]'`. After pushing this repository update to GitHub, use the staged instructions in [docs/huggingface/publish.md](docs/huggingface/publish.md) to authenticate and upload the pretrained encoder bundle from Colab. Contributions and replication details are in [CONTRIBUTING.md](CONTRIBUTING.md). No model license is set until the source dataset rights are verified.

For Colab-ready matched probe commands and plots of pretraining/probe results, see [docs/colab_probe_visualization.md](docs/colab_probe_visualization.md). In Colab, run shell commands in a `%%bash` cell and plotting snippets in normal Python cells.

For the complete staged real-data workflow with a live Colab snapshot monitor and scratch-versus-pretrained transfer comparison, see [docs/colab_training_stages.md](docs/colab_training_stages.md).
