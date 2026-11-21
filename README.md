# PIXREL

PIXREL (Pixel Relationship Learning) is an experimental computer vision research project. This repository starts with a reproducible implementation of a pixel relationship encoder and the controls needed to test whether adaptive local relationships add value over convolution or fixed neighborhood aggregation. Training can run on CPU or CUDA GPUs.

No superiority or foundation-model claims are made. Reported metrics are limited to the specific datasets, protocols, and runs documented in the experiment reports.

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

The initial pretraining command uses unlabeled images from the 45,000-image CIFAR-100 training subset. It predicts clean teacher features from masked student inputs at H0, at the fused spatial output, and at the global H3 embedding. CIFAR-100 is only a pipeline and representation-learning pilot, not a foundation-model corpus.

```bash
pixrel-pretrain --variant adaptive --seed 17 --epochs 100 --batch-size 128 --device cuda --data ./data --output ./outputs/pretrain-cifar100
```

The encoder weights are written to `encoder.pt`; `last.pt` includes teacher, predictors, optimizer, scheduler, and RNG state for resume. After pretraining, run a frozen linear probe:

```bash
pixrel-train --variant adaptive --seed 17 --epochs 20 --batch-size 64 --device cuda --init-encoder ./outputs/pretrain-cifar100/encoder.pt --freeze-encoder --data ./data --output ./outputs/probe-cifar100-seed17
```

For a random-frozen-feature control, use the same command without `--init-encoder` and save to a separate output directory. For end-to-end fine-tuning, pass `--init-encoder` and omit `--freeze-encoder`.

See [docs/foundation_model_plan.md](docs/foundation_model_plan.md) before interpreting this pilot or scaling to a larger image corpus.
