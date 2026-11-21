# Colab training stages: masked pixels to downstream task

**Completed run status:** the fire-image workflow in this guide was run through the 20-epoch pretraining stage and the three-seed transfer/test comparison. Pretrained averaged 94.08% test accuracy versus 95.83% for scratch; see the [final report](final_report.md) before reusing the sample commands.

This workflow validates the pipeline, trains on permitted images, watches pixel and feature learning, and compares downstream classification with a same-architecture scratch baseline. RGB reconstruction is a visible training objective, not proof of general vision ability. The downstream task measures transfer.

## Stage 0 — Update Colab

From your development machine, publish the repository changes to GitHub, then mount Drive in a normal Python cell:

~~~bash
git add README.md docs pixrel pyproject.toml
git commit -m "Add RGB reconstruction and live pixel visualizations"
git push
~~~

~~~python
from google.colab import drive
drive.mount("/content/drive")
~~~

Then pull and install in a shell cell. %%bash must be the first line:

~~~bash
%%bash
set -e
cd /content/Pixrel
git pull
pip install -e .
~~~

If the repository is not cloned into this runtime yet, clone it to /content/Pixrel first.

## Stage 1 — Prepare image folders

Unlabeled pretraining scans the chosen directory recursively; images can be flat or nested, and folder names are ignored. The directory must exist and contain supported image files:

~~~text
/content/unlabeled-images/
├── image-0001.jpg
├── image-0002.png
└── more-images/
    └── image-0003.jpg
~~~

For downstream classification, keep splits separate and use one class directory per label:

~~~text
/content/my-task/
├── train/class_a/*.jpg
├── train/class_b/*.jpg
├── val/class_a/*.jpg
├── val/class_b/*.jpg
├── test/class_a/*.jpg
└── test/class_b/*.jpg
~~~

Record each dataset's source, version, usage terms, filtering, and split.

## Stage 2 — Smoke-run on real images

Change the data path to your mounted image folder. This runs 10 batches and writes a visual snapshot:

~~~bash
%%bash
set -e
pixrel-pretrain --dataset imagefolder --variant adaptive --seed 17 \
  --epochs 1 --limit-train-batches 10 --image-size 128 --patch-size 4 \
  --batch-size 4 --rgb-weight 0.1 --visualize-every 1 --device cuda \
  --data /content/unlabeled-images \
  --output /content/drive/MyDrive/pixrel-runs/pixrel-smoke
~~~

Expect encoder_bundle.pt, history.json, pretrain.json, and visualizations/ with epoch_0001.png, feature_path_progress.gif, and learning_curves.png. This smoke run checks the path only; it is not a model result.

View the smoke artifacts in a normal Python cell:

~~~python
from IPython.display import Image, display
from pathlib import Path

viz = Path("/content/drive/MyDrive/pixrel-runs/pixrel-smoke/visualizations")
display(Image(filename=str(viz / "epoch_0001.png")))
display(Image(filename=str(viz / "learning_curves.png")))
~~~

Snapshots show the input and mask, H0–H3 RGB reconstructions, masked-pixel error, teacher/student local features, stage activation maps, fused maps, and one center pixel's learned 3×3 neighbor weights. Curves include masked RGB MAE and PSNR. Feature maps are normalized to show patterns; compare numeric MAE/PSNR for reconstruction trends.

## Stage 3 — Train and watch the full run

Run this in a normal Python cell. It launches training, logs console output to Drive, and refreshes the latest snapshot as it appears. Image size 128 and batch 16 are starting values for an L4; lower the batch if GPU memory runs out.

~~~python
from pathlib import Path
from IPython.display import Image, clear_output, display
import subprocess
import time

repo = Path("/content/Pixrel")
data = Path("/content/unlabeled-images")  # change to your image directory
output = Path("/content/drive/MyDrive/pixrel-runs/pixrel-imagefolder-128")
viz = output / "visualizations"
output.mkdir(parents=True, exist_ok=True)
log_path = output / "train.log"

command = [
    "pixrel-pretrain", "--dataset", "imagefolder", "--variant", "adaptive",
    "--seed", "17", "--epochs", "100", "--image-size", "128", "--patch-size", "4",
    "--batch-size", "16", "--rgb-weight", "0.1", "--visualize-every", "2",
    "--device", "cuda", "--data", str(data), "--output", str(output),
]

last_snapshot = None
last_log_size = -1
with log_path.open("w") as log:
    process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, text=True)
    while process.poll() is None:
        snapshots = sorted(viz.glob("epoch_*.png"))
        latest = snapshots[-1] if snapshots else None
        log_size = log_path.stat().st_size if log_path.exists() else 0
        if latest != last_snapshot or log_size != last_log_size:
            clear_output(wait=True)
            print(f"Training process running. Log: {log_path}")
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-12:]))
            if latest:
                display(Image(filename=str(latest), width=1400))
                curves = viz / "learning_curves.png"
                if curves.exists():
                    display(Image(filename=str(curves), width=1100))
            last_snapshot = latest
            last_log_size = log_size
        time.sleep(10)
    return_code = process.wait()

if return_code != 0:
    print("\n".join(log_path.read_text(errors="replace").splitlines()[-60:]))
    raise RuntimeError(f"pixrel-pretrain exited with status {return_code}")
print("Pretraining finished:", output)
~~~

For out-of-memory errors, reduce batch size first. Image size must be divisible by patch size. Resume with the matching last.pt and same core settings. Checkpoints from before RGB reconstruction cannot resume in this version.

## Stage 4 — Compare transfer against scratch

Use identical task folders, seeds, and training settings:

~~~bash
%%bash
set -e
RUNS=/content/drive/MyDrive/pixrel-runs
TASK=/content/my-task

for seed in 17 29 43; do
  pixrel-transfer --checkpoint "$RUNS/pixrel-imagefolder-128/encoder_bundle.pt" \
    --train-dir "$TASK/train" --val-dir "$TASK/val" --seed "$seed" \
    --epochs 30 --batch-size 32 --device cuda --output "$RUNS/task-pretrained-$seed"

  pixrel-transfer --scratch --variant adaptive --image-size 128 \
    --train-dir "$TASK/train" --val-dir "$TASK/val" --seed "$seed" \
    --epochs 30 --batch-size 32 --device cuda --output "$RUNS/task-scratch-$seed"
done
~~~

Choose settings from validation only. Then evaluate the selected checkpoints on the separate test folder once:

~~~bash
%%bash
set -e
RUNS=/content/drive/MyDrive/pixrel-runs
TASK=/content/my-task

for seed in 17 29 43; do
  pixrel-transfer-evaluate --checkpoint "$RUNS/task-pretrained-$seed/best.pt" \
    --data-dir "$TASK/test" --device cuda --output "$RUNS/task-pretrained-$seed/test.json"
  pixrel-transfer-evaluate --checkpoint "$RUNS/task-scratch-$seed/best.pt" \
    --data-dir "$TASK/test" --device cuda --output "$RUNS/task-scratch-$seed/test.json"
done
~~~

The current transfer CLI supports classification. Segmentation and other dense tasks need their own heads and evaluation before claims of transfer.
