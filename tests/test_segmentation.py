import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from pixrel.models.encoder import PixrelEncoder
from pixrel.models.segmentation import PixrelSegmenter
from pixrel.segmentation import PairedMaskDataset, _run_epoch, _train, segmentation_metrics


class PixrelSegmentationTests(unittest.TestCase):
    def test_segmenter_returns_full_resolution_logits_and_gradients(self):
        torch.set_num_threads(1)
        encoder = PixrelEncoder(num_classes=2, variant="conv", widths=(8, 8, 8, 8))
        model = PixrelSegmenter(encoder, num_classes=3, decoder_channels=8)
        images = torch.randn(2, 3, 16, 16)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 3, 16, 16))
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 3, (2, 16, 16)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.encoder.fuse[0].weight.grad)
        self.assertIsNotNone(model.decoder[-1].weight.grad)

    def test_paired_dataset_resizes_masks_nearest_and_preserves_class_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images_dir = root / "images"
            masks_dir = root / "masks"
            images_dir.mkdir()
            masks_dir.mkdir()
            Image.fromarray(np.full((8, 10, 3), 127, dtype=np.uint8), mode="RGB").save(images_dir / "sample.v1.jpg")
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[:, 5:] = 1
            Image.fromarray(mask, mode="L").save(masks_dir / "sample.v1.png")

            dataset = PairedMaskDataset(images_dir, masks_dir, image_size=16, num_classes=2)
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(tuple(target.shape), (16, 16))
            self.assertEqual(set(torch.unique(target).tolist()), {0, 1})

    def test_metrics_and_training_epoch(self):
        confusion = torch.tensor([[2, 1], [1, 3]])
        metrics = segmentation_metrics(confusion, ["background", "object"])
        self.assertAlmostEqual(metrics["mean_iou"], 0.55)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 5 / 7)
        self.assertEqual(metrics["per_class_iou"], {"background": 0.5, "object": 0.6})

        torch.set_num_threads(1)
        model = PixrelSegmenter(
            PixrelEncoder(num_classes=2, variant="conv", widths=(8, 8, 8, 8)),
            num_classes=2,
            decoder_channels=8,
        )
        images = torch.randn(2, 3, 16, 16)
        masks = torch.randint(0, 2, (2, 16, 16))
        loader = DataLoader(list(zip(images, masks)), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = _run_epoch(model, loader, torch.device("cpu"), 2, 255, optimizer)
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["valid_pixels"], 512)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertGreaterEqual(result["mean_iou"], 0.0)

    def test_one_epoch_training_writes_reloadable_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                images_dir = root / split / "images"
                masks_dir = root / split / "masks"
                images_dir.mkdir(parents=True)
                masks_dir.mkdir(parents=True)
                for index in range(2):
                    image = np.full((16, 16, 3), 80 + index * 40, dtype=np.uint8)
                    mask = np.zeros((16, 16), dtype=np.uint8)
                    mask[:, 8:] = 1
                    Image.fromarray(image, mode="RGB").save(images_dir / f"sample{index}.jpg")
                    Image.fromarray(mask, mode="L").save(masks_dir / f"sample{index}.png")
            output = root / "run"
            args = SimpleNamespace(
                seed=3, threads=1, device="cpu", scratch=True, image_size=16,
                variant="conv", num_classes=2, checkpoint=None,
                train_images=str(root / "train/images"), train_masks=str(root / "train/masks"),
                val_images=str(root / "val/images"), val_masks=str(root / "val/masks"),
                ignore_index=255, class_names=["background", "foreground"],
                batch_size=2, workers=0, freeze_encoder=False, lr=1e-3,
                weight_decay=0.0, epochs=1, output=str(output),
            )
            _train(args)
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            encoder = PixrelEncoder(num_classes=2, variant=checkpoint["variant"],
                                    widths=tuple(checkpoint["widths"]))
            model = PixrelSegmenter(encoder, num_classes=2)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            with torch.inference_mode():
                logits = model(torch.randn(1, 3, 16, 16))
            self.assertEqual(tuple(logits.shape), (1, 2, 16, 16))


if __name__ == "__main__":
    unittest.main()
