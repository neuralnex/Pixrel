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

## 2. Install and enable a Colab-native drawing canvas

Use `ipycanvas` for drag painting. This avoids switching Matplotlib backends, which can fail in Colab with `ValueError: Key backend ... ipympl.backend_nbagg`. Run setup once; if Colab reports that the widget model is unavailable, restart the runtime and rerun this cell.

```python
%pip -q install ipycanvas
from google.colab import output
output.enable_custom_widget_manager()
```

## 3. Paint flame pixels and save masks

In the cell below, choose **Paint flame**, then hold the left mouse button and drag over visible flames. Choose **Erase** to correct mistakes; adjust the brush size as needed. Click **Save flame mask + next** when finished. If the image contains no visible flame, explicitly click **Save no-flame mask + next**. Skipped images remain pending for later review. Everything not painted is background (`0`).

```python
from pathlib import Path
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from PIL import Image, ImageDraw
from ipycanvas import MultiCanvas, hold_canvas

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
    state = {"index": 0, "image": None, "mask": None, "painting": False,
             "previous": None, "canvas_size": None}
    status = widgets.HTML()
    mode = widgets.ToggleButtons(
        options=[("Paint flame", 1), ("Erase", 0)], value=1
    )
    brush_size = widgets.IntSlider(value=8, min=1, max=32, step=1, description="Brush")
    save_button = widgets.Button(description="Save flame mask + next", button_style="success")
    no_flame_button = widgets.Button(description="Save no-flame mask + next")
    skip_button = widgets.Button(description="Skip for review")
    # Layer 0 shows the source image; layer 1 shows translucent flame pixels.
    canvas = MultiCanvas(2, width=640, height=480, sync_image_data=True)
    canvas.layout.width = "min(100%, 640px)"
    image_layer, mask_layer = canvas[0], canvas[1]

    def paint_segment(start, end, value):
        if start is None:
            return
        x0, y0 = start
        x1, y1 = end
        radius = int(brush_size.value)
        color = "rgba(255, 35, 10, 0.62)" if value else "rgba(0, 0, 0, 0)"
        with hold_canvas(mask_layer):
            if value == 0:
                # Clear painted pixels beneath the eraser using the canvas API.
                mask_layer.global_composite_operation = "destination-out"
            else:
                mask_layer.global_composite_operation = "source-over"
            mask_layer.stroke_style = color
            mask_layer.fill_style = color
            mask_layer.line_width = 2 * radius
            mask_layer.line_cap = "round"
            mask_layer.line_join = "round"
            mask_layer.begin_path()
            mask_layer.move_to(x0, y0)
            mask_layer.line_to(x1, y1)
            mask_layer.stroke()
            mask_layer.begin_path()
            mask_layer.arc(x1, y1, radius, 0, 2 * np.pi)
            mask_layer.fill()
            mask_layer.global_composite_operation = "source-over"

        # Store the same brush path in the binary mask at original image resolution.
        scale_x = state["mask"].shape[1] / state["canvas_size"][0]
        scale_y = state["mask"].shape[0] / state["canvas_size"][1]
        draw = ImageDraw.Draw(state["mask"])
        xy = (round(x0 * scale_x), round(y0 * scale_y),
              round(x1 * scale_x), round(y1 * scale_y))
        width = max(1, round(2 * radius * (scale_x + scale_y) / 2))
        if value:
            draw.line(xy, fill=1, width=width)
            r = max(1, width // 2)
            ex, ey = xy[2], xy[3]
            draw.ellipse((ex-r, ey-r, ex+r, ey+r), fill=1)
        else:
            # Use a separate binary draw operation below for erasing.
            draw.line(xy, fill=0, width=width)
            r = max(1, width // 2)
            ex, ey = xy[2], xy[3]
            draw.ellipse((ex-r, ey-r, ex+r, ey+r), fill=0)

    def show_current():
        split, image_path, _ = pending[state["index"]]
        state["image"] = Image.open(image_path).convert("RGB")
        width, height = state["image"].size
        state["mask"] = Image.new("L", (width, height), 0)
        state["painting"] = False
        state["previous"] = None
        scale = min(640 / width, 480 / height, 1.0)
        canvas_width, canvas_height = max(1, round(width * scale)), max(1, round(height * scale))
        canvas.width, canvas.height = canvas_width, canvas_height
        state["canvas_size"] = (canvas_width, canvas_height)
        image_layer.clear()
        mask_layer.clear()
        preview = state["image"].resize((canvas_width, canvas_height))
        image_layer.put_image_data(np.asarray(preview, dtype=np.uint8), 0, 0)
        status.value = f"Split: <b>{split}</b> · class ID 1 = visible flame"
        status.value += "<br>Drag on the canvas to paint; eraser removes strokes."

    def on_mouse_down(x, y):
        state["painting"] = True
        state["previous"] = (x, y)
        paint_segment((x, y), (x, y), int(mode.value))

    def on_mouse_move(x, y):
        if state["painting"]:
            point = (x, y)
            paint_segment(state["previous"], point, int(mode.value))
            state["previous"] = point

    def on_mouse_up(x, y):
        state["painting"] = False
        state["previous"] = None

    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_move(on_mouse_move)
    canvas.on_mouse_up(on_mouse_up)

    def advance():
        state["index"] += 1
        if state["index"] >= len(pending):
            image_layer.clear()
            mask_layer.clear()
            status.value = "Finished pending images. Review mask overlays before training."
        else:
            show_current()

    def on_save(_):
        mask_array = np.asarray(state["mask"], dtype=np.uint8)
        if not np.any(mask_array == 1):
            status.value = "No flame pixels painted. Use Save no-flame if there is no visible flame."
            return
        split, image_path, mask_path = pending[state["index"]]
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        state["mask"].save(mask_path)
        print(f"Saved {split}/{image_path.name} -> {mask_path}")
        advance()

    def on_save_no_flame(_):
        split, image_path, mask_path = pending[state["index"]]
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        state["mask"].save(mask_path)
        print(f"Saved reviewed no-flame mask for {split}/{image_path.name}")
        advance()

    def on_skip(_):
        state["index"] += 1
        if state["index"] >= len(pending):
            image_layer.clear()
            mask_layer.clear()
            status.value = "Finished pending list; skipped images still need annotation or exclusion."
        else:
            show_current()

    save_button.on_click(on_save)
    no_flame_button.on_click(on_save_no_flame)
    skip_button.on_click(on_skip)
    display(widgets.VBox([
        canvas,
        widgets.HBox([mode, brush_size]),
        widgets.HBox([save_button, no_flame_button, skip_button]),
        status,
    ]))
    show_current()
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
