# Colab: probe runs and visualizations

The CIFAR-100 probe workflow and completed results are summarized in the [final report](final_report.md) and the detailed [CIFAR-100 SSL report](experiments/cifar100-ssl-seed17.md). The fire-image transfer outcome is separate and does not show a pretrained advantage.

The training command is shell code. A normal Colab cell is Python, so a line like `RUNS=/content/...` raises `SyntaxError`. Use a `%%bash` cell for shell commands (the cell magic must be the first line), or prefix individual shell commands with `!`. Use a normal Python cell for plotting.

## 1. Run matched frozen probes

Run this in a Colab cell after mounting Drive, changing to the repository, and installing PIXREL. These runs select checkpoints using validation accuracy; do not use the test set to choose settings.

```bash
%%bash
set -e
RUNS=/content/drive/MyDrive/pixrel-runs
DATA=/content/pixrel-data

for seed in 17 29 43; do
  pixrel-train --variant adaptive --seed "$seed" --epochs 20 --batch-size 64 --device cuda \
    --init-encoder "$RUNS/pretrain-cifar100/encoder.pt" --freeze-encoder \
    --data "$DATA" --output "$RUNS/probe-pretrained-seed$seed"

  pixrel-train --variant adaptive --seed "$seed" --epochs 20 --batch-size 64 --device cuda \
    --freeze-encoder \
    --data "$DATA" --output "$RUNS/probe-random-seed$seed"
done
```

If the notebook rejects `%%bash`, make sure it is the very first line in the cell, with no preceding spaces or Python code. To set paths in a Python cell instead, use `RUNS = "/content/drive/MyDrive/pixrel-runs"` and `DATA = "/content/pixrel-data"`, then invoke commands as `!pixrel-train ...`.

## 2. View pretraining diagnostics and the pixel-feature path

New pretraining runs automatically create snapshots and a GIF under `<output>/visualizations/` every `--visualize-every` epochs (default 5), plus the first and final epoch. In a normal Python cell, display the progress animation and loss curves:

```python
from IPython.display import Image, display
from pathlib import Path

viz_dir = Path("/content/drive/MyDrive/pixrel-runs/pretrain-cifar100/visualizations")
display(Image(filename=str(viz_dir / "feature_path_progress.gif")))
display(Image(filename=str(viz_dir / "learning_curves.png")))
```

In the reconstruction-enabled trainer, each epoch image also includes H0–H3 RGB reconstructions and masked-pixel error; the curves include masked RGB MAE and PSNR. This shows what the model predicts for hidden pixels plus feature activations and local affinities. It is a diagnostic view, not proof that a path is causal or that the features transfer to a task. Maps are individually normalized to emphasize shape, so compare numeric MAE/PSNR rather than color intensity across epochs. Earlier checkpoints created before the RGB objective do not contain those reconstructions.

The following cell can also be used to plot pretraining diagnostics from `history.json`:

This reads the `history.json` written beside `encoder.pt` and plots the logged losses and feature standard deviations. It saves a PNG in Drive and displays it in the notebook.

```python
from pathlib import Path
import json
import matplotlib.pyplot as plt

pretrain_dir = Path("/content/drive/MyDrive/pixrel-runs/pretrain-cifar100")
history = json.loads((pretrain_dir / "history.json").read_text())
epochs = [row["epoch"] for row in history]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for key, label in [
    ("total_loss", "total"),
    ("local_pixel_cosine_loss", "local pixel"),
    ("fused_spatial_cosine_loss", "fused spatial"),
    ("global_cosine_loss", "global"),
]:
    axes[0].plot(epochs, [row[key] for row in history], label=label)
axes[0].set(title="PIXREL pretraining losses", xlabel="Epoch", ylabel="Cosine loss")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(epochs, [row["teacher_feature_std"] for row in history], label="local teacher feature std")
axes[1].plot(epochs, [row["teacher_embedding_channel_std"] for row in history], label="global embedding channel std")
axes[1].set(title="Feature variation diagnostics", xlabel="Epoch", ylabel="Standard deviation")
axes[1].grid(alpha=0.25)
axes[1].legend()

fig.tight_layout()
plot_path = pretrain_dir / "pretraining_diagnostics.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path}")
```

## 3. Plot pretrained versus random frozen-probe validation accuracy

This uses each run's `run.json` and plots all three seeds plus mean ± standard deviation. The metric is the best validation accuracy recorded during classifier training.

```python
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

runs = Path("/content/drive/MyDrive/pixrel-runs")
seeds = [17, 29, 43]
groups = {
    "Pretrained frozen": [
        json.loads((runs / f"probe-pretrained-seed{s}" / "run.json").read_text())["best_val_accuracy"]
        for s in seeds
    ],
    "Random frozen": [
        json.loads((runs / f"probe-random-seed{s}" / "run.json").read_text())["best_val_accuracy"]
        for s in seeds
    ],
}

fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(groups))
means = [np.mean(values) for values in groups.values()]
stds = [np.std(values, ddof=1) for values in groups.values()]
ax.bar(xs, means, yerr=stds, capsize=7, alpha=0.75)
for i, (label, values) in enumerate(groups.items()):
    jitter = np.linspace(-0.08, 0.08, len(values))
    ax.scatter(xs[i] + jitter, values, color="black", zorder=3, label="seed result" if i == 0 else None)
    print(f"{label}: " + ", ".join(f"seed {seed}={value:.4f}" for seed, value in zip(seeds, values)))
    print(f"  mean ± sample SD: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")

ax.set_xticks(xs, list(groups))
ax.set_ylabel("Best validation accuracy")
ax.set_title("Frozen-feature probe: pretrained vs random encoder")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
plot_path = runs / "frozen_probe_comparison.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path}")
```

The documented CIFAR-100 probe and official test runs are complete: pretrained features beat random frozen features across all three seeds, with low absolute accuracy. Do not rerun or tune against that test split. For current project results and next evidence gates, see the [final report](final_report.md).
