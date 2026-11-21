# PIXREL

PIXREL (Pixel Relationship Learning) is an experimental computer vision research project. This repository starts with a reproducible, CPU-capable implementation of a pixel relationship encoder and the controls needed to test whether adaptive local relationships add value over convolution or fixed neighborhood aggregation.

No performance or foundation-model claims are made. A model becomes meaningful only after real-data training and controlled evaluation.

## Quick start

Use Python 3.10+ and install the package and dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Run a CPU CIFAR-100 experiment (downloads the dataset on first run):

```bash
pixrel-train --variant adaptive --seed 17 --epochs 20 --data ./data --output ./outputs/adaptive-seed17
pixrel-evaluate --checkpoint ./outputs/adaptive-seed17/best.pt --data ./data --output ./outputs/adaptive-seed17/eval.json
```

Variants are `adaptive`, `fixed`, and `conv`. Run each with the same seeds and training settings. CPU runtime can be substantial; begin with `--epochs 1 --limit-train-batches 10 --limit-val-batches 5` as a pipeline smoke run. These limited runs are not benchmark results. Use `--threads` to control CPU parallelism. The code selects CPU explicitly and does not initialize CUDA.

The CIFAR download progress reaching `100%` only marks the download/extraction step; training follows afterward. The script now prints when data is ready, when each epoch begins, and a status line every 10 training batches. The adaptive model is compute intensive on CPU, so a full epoch can take a while.

## Research protocol

The first benchmark compares the three variants on CIFAR-100 using identical data splits, augmentation, optimizer, schedule, epochs, and predetermined seeds (17, 29, 43). The training set is split into train/validation; the official test set is evaluated only after model selection. Report mean and standard deviation across seeds, parameter counts, latency, and the full run configuration. A short run validates engineering only. Do not tune on the test set.

The controls share a hierarchical encoder skeleton. `adaptive` learns content-dependent neighbor weights, `fixed` uses uniform valid-neighbor weights, and `conv` uses a local convolution block. These are initial controls, not guaranteed parameter/FLOP-matched baselines. Record parameter counts and computational costs before interpreting differences.

See [docs/architecture.md](docs/architecture.md), [docs/benchmark_protocol.md](docs/benchmark_protocol.md), and [docs/research_blueprint.md](docs/research_blueprint.md).
