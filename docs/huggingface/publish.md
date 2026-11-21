# Publish the PIXREL fire-transfer checkpoint to Hugging Face

The initial Hub artifact is the **pretrained encoder bundle**, not the fire classifier. Contributors can download it and fine-tune it for their own labeled task. The benchmark shows the scratch baseline ahead on average, so describe this as an experimental encoder and report that result with the checkpoint.

See the [final report](../final_report.md) for the full metrics, caveats, and current publication status.

## 1. Make the uploader available in Colab

Commit and push the repository changes to the existing GitHub repository. In Colab, pull and install the Hub extra:

```bash
%%bash
set -e
cd /content/Pixrel
git pull
pip install -e '.[hub]'
```

## 2. Authenticate to Hugging Face

Use a Hugging Face account with permission to create a model repository. Run in a normal Python cell; do not paste a token into a notebook cell or save it in Drive:

```python
from huggingface_hub import notebook_login
notebook_login()
```

## 3. Upload the encoder and selected evidence

The model repository is `nexusbert/pixrel-fire-encoder-v1`. This creates a **public** model repository by default. The uploader copies the model card to the repository root as `README.md`, alongside the checkpoint, pretraining summary, and selected visualization artifacts. It does not upload source images, transfer checkpoints, training logs, or tokens.

```bash
%%bash
set -e

RUN="/content/drive/MyDrive/pixrel-runs/fire-pretrain-train-only-seed17"
pixrel-hub-upload \
  --repo-id "nexusbert/pixrel-fire-encoder-v1" \
  --checkpoint "$RUN/encoder_bundle.pt" \
  --model-card /content/Pixrel/model_cards/fire-detection-seed17/README.md \
  --pretrain-summary "$RUN/pretrain.json" \
  --visualizations-dir "$RUN/visualizations"
```

Add `--private` if you want to stage it privately first. Confirm the model page contains `encoder_bundle.pt`, `README.md`, the correct benchmark results, and the intended visibility before sharing the link. The model card currently has no license field because the dataset's reuse and derivative-weight terms have not been verified; do not claim a license until those rights are checked.

## 4. Download and fine-tune

Users can download the bundle with `huggingface_hub` and then use the PIXREL transfer CLI:

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="nexusbert/pixrel-fire-encoder-v1",
    filename="encoder_bundle.pt",
)
print(checkpoint)
```

```bash
pixrel-transfer --checkpoint /path/printed/above/encoder_bundle.pt \
  --train-dir /path/to/data/train --val-dir /path/to/data/val \
  --epochs 30 --batch-size 32 --device auto --output ./outputs/my-task
```

The task folders need one subdirectory per class. Users should preserve a held-out test set, compare with a same-architecture scratch baseline, and report dataset terms, split construction, seeds, and per-class metrics. See [CONTRIBUTING.md](../../CONTRIBUTING.md).
