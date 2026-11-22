# Colab: create a 200-image fire segmentation set before training

This workflow creates real binary masks by manually outlining visible flames. It performs no GPU training. The non-fire class receives all-background masks based on its image-level label. Fire pixels are class ID `1`; everything else, including smoke for this first experiment, is class ID `0`.

This is a flame-localization task, not a general fire/smoke detector. Review every annotated mask. Do not use automatic or guessed masks as ground truth.

## 1. Mount Drive and create a balanced 200-image subset

The source must be the existing split at `/content/fire-benchmark-v1` with `train`, `val`, and `test` directories and class folders `fire_images` and `non_fire_images`. It selects 70 images per class for train, 15 per class for validation, and 15 per class for test. This preserves the earlier split, so validation/test images stay out of the pretraining data.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path
import random
import shutil
import numpy as np
from PIL import Image

source = Path("/content/fire-benchmark-v1")
destination = Path("/content/drive/MyDrive/pixrel-fire-seg-200")
classes = ("fire_images", "non_fire_images")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not source.is_dir():
    raise FileNotFoundError(f"Restore the existing split first: {source}")
if destination.exists():
    raise FileExistsError(f"Destination already exists: {destination}")

rng = random.Random(17)
for split, count in (("train", 70), ("val", 15), ("test", 15)):
    for class_name in classes:
        source_dir = source / split / class_name
        files = sorted(
            path for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if len(files) < count:
            raise ValueError(f"Need {count} images in {source_dir}; found {len(files)}")
        selected = rng.sample(files, count)
        for index, path in enumerate(selected):
            stem = f"{class_name}/{index:04d}_{path.stem}"
            image_target = destination / split / "images" / Path(stem + path.suffix.lower())
            mask_target = destination / split / "masks" / Path(stem + ".png")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, image_target)

            # Non-fire image labels imply no visible flame for this binary task.
            if class_name == "non_fire_images":
                with Image.open(path) as image:
                    width, height = image.size
                Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L").save(mask_target)

        print(f"{split}/{class_name}: copied {count} images")

print("Subset and non-fire masks created:", destination)
```

## 2. Install and enable interactive polygon annotation

Run this in a Colab cell. It uses an interactive Matplotlib canvas; if the canvas does not become clickable after installation, restart the Colab runtime and rerun the setup cell.

```python
%pip -q install ipympl
from google.colab import output
output.enable_custom_widget_manager()
%matplotlib widget
```

## 3. Outline flames and save masks

In the cell below, each left-click adds a polygon vertex; double-click the final vertex to finish that flame region. Draw separate polygons for disconnected flame regions. Click **Save mask + next** when all visible flames in the current image are outlined. The rest of the mask remains background (`0`). Images without a visible flame should be skipped for review, not saved as an all-background fire mask.

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw

ROOT = Path("/content/drive/MyDrive/pixrel-fire-seg-200")
# Build exact image->mask paths, independent of Colab's mount path depth.
pending = []
for split in ("train", "val", "test"):
    for image in sorted((ROOT / split / "images" / "fire_images").glob("*")):
        mask = ROOT / split / "masks" / "fire_images" / f"{image.stem}.png"
        if not mask.exists():
            pending.append((split, image, mask))

if not pending:
    print("All fire images already have masks.")
else:
    state = {"index": 0, "polygons": [], "selector": None, "image": None}
    fig, ax = plt.subplots(figsize=(8, 7))
    status = widgets.HTML()
    save_button = widgets.Button(description="Save mask + next", button_style="success")
    skip_button = widgets.Button(description="Skip for review")

    def show_current():
        if state["selector"] is not None:
            state["selector"].disconnect_events()
        split, image_path, mask_path = pending[state["index"]]
        state["image"] = np.asarray(Image.open(image_path).convert("RGB"))
        state["polygons"] = []
        ax.clear()
        ax.imshow(state["image"])
        ax.set_title(f"{state['index'] + 1}/{len(pending)}  {split}/{image_path.name}\n"
                     "Outline visible flame regions; double-click to finish each polygon")
        ax.axis("off")

        def on_polygon(vertices):
            state["polygons"].append(np.asarray(vertices, dtype=np.float32))
            ax.add_patch(Polygon(vertices, closed=True, fill=False, edgecolor="yellow", linewidth=2))
            fig.canvas.draw_idle()

        state["selector"] = PolygonSelector(ax, on_polygon, useblit=True)
        status.value = f"Split: <b>{split}</b> · polygons drawn: {len(state['polygons'])}"
        fig.canvas.draw_idle()

    def on_save(_):
        if not state["polygons"]:
            status.value = "Draw at least one flame polygon, or skip this image for review."
            return
        split, image_path, mask_path = pending[state["index"]]
        height, width = state["image"].shape[:2]
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        for polygon in state["polygons"]:
            points = [(int(round(x)), int(round(y))) for x, y in polygon]
            draw.polygon(points, fill=1)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_image.save(mask_path)
        print(f"Saved {split}/{image_path.name} -> {mask_path}")
        state["index"] += 1
        if state["index"] >= len(pending):
            if state["selector"] is not None:
                state["selector"].disconnect_events()
            ax.clear()
            ax.axis("off")
            status.value = "Finished all pending fire images. Review mask overlays before training."
            fig.canvas.draw_idle()
        else:
            show_current()

    def on_skip(_):
        state["index"] += 1
        if state["index"] >= len(pending):
            if state["selector"] is not None:
                state["selector"].disconnect_events()
            ax.clear()
            ax.axis("off")
            status.value = "Finished pending list; skipped images still need annotation or exclusion."
            fig.canvas.draw_idle()
        else:
            show_current()

    save_button.on_click(on_save)
    skip_button.on_click(on_skip)
    display(widgets.HBox([save_button, skip_button]), status)
    show_current()
    plt.show()
```

If a fire image has no visible flame, the current annotation cell's skip button leaves it without a mask; either exclude it from this 200-image experiment or intentionally create an all-background mask after reviewing it. Do not let it remain missing when starting training.

## 4. Review masks before training

Run this cell to overlay a mask on its source image. Confirm yellow/white pixels cover flames only. Repeat for several images in every split, and inspect class counts.

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root = Path("/content/drive/MyDrive/pixrel-fire-seg-200")
image_path = next((root / "train/images/fire_images").glob("*"))
mask_path = root / "train/masks/fire_images" / f"{image_path.stem}.png"

image = np.asarray(Image.open(image_path).convert("RGB"))
mask = np.asarray(Image.open(mask_path))
overlay = image.copy()
overlay[mask == 1] = (255, 40, 20)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(image); plt.title("Image"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(overlay); plt.title("Red = flame mask"); plt.axis("off")
plt.show()

for split in ("train", "val", "test"):
    masks = list((root / split / "masks").rglob("*.png"))
    print(split, "masks:", len(masks))
```

Expected mask count is 140 train, 30 validation, and 30 test. Resolve skipped/missing masks before training. The current PIXREL segmentation trainer uses square resize and nearest-neighbor mask resizing; document that preprocessing in any report.

## 5. Train only after reviewing annotations

After the masks are complete and checked, follow the train/evaluate commands in [the segmentation guide](segmentation.md). Compare pretrained and scratch using identical splits and settings. With only 30 test images, report per-image counts and treat the result as a pilot.
