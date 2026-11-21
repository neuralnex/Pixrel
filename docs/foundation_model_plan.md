# PIXREL foundation-model research plan

## What the term means here

The target is a reusable visual encoder pretrained on diverse, permitted images without relying on downstream labels, then shown to transfer to multiple downstream tasks. The current CIFAR-100 supervised benchmark is not a foundation model. The new CIFAR self-supervised path is an engineering and representation pilot only.

## Current pilot objective

The student receives a CIFAR image with random 4×4-aligned patches replaced by zero in normalized input space. The EMA teacher receives the unmasked image. A student predictor learns normalized teacher targets at three levels:

1. H0 local pixel features, evaluated only at masked positions.
2. The fused high-resolution spatial feature map, evaluated only at masked positions.
3. The H3-pooled global image embedding.

The total loss is local pixel cosine loss + fused spatial cosine loss + `global_weight` times global cosine loss. The teacher encoder is an EMA of the online encoder. The prototype logs loss components and feature-variance diagnostics, writes encoder weights and resumable state, and deliberately does not add an unvalidated collapse-prevention loss.

This is a testable pilot design, not a claim of novelty or a validated pretraining recipe. Its choices should be compared with established approaches. Masked autoencoders reconstruct masked pixels with a visible-patch encoder and decoder ([He et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)); I-JEPA predicts target representations in embedding space ([Assran et al., 2023](https://arxiv.org/abs/2301.08243)); DINOv2 reports robust features from self-supervision with sufficiently large, curated, diverse data ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)). These are reference points, not methods PIXREL has matched or surpassed.

## Staged gates

### FM-0 — Fair encoder evidence

Complete the capacity-aware, longer CIFAR comparison described in [Relationship Value Benchmark V1](experiments/relationship-value-v1.md). The convolution baseline currently leads the 20-epoch pilot.

### FM-1 — Self-supervised engineering

Run a 1-epoch, limited-batch GPU check to validate masking, teacher updates, checkpoint writes, and resume. Then run the full CIFAR pilot and inspect both loss curves and embedding variance. A decreasing loss alone is not proof of useful features.

### FM-2 — Representation evaluation

Run frozen linear probes with the pretrained encoder and a same-architecture random-init control, using the same seed and data split. Then compare fine-tuning. Keep the official test split out of choices about architecture, schedule, and loss weights. CIFAR-100 results are in-domain representation evidence, not cross-domain transfer.

The CLI supports both probe controls. For the pretrained frozen probe, pass `--init-encoder <encoder.pt> --freeze-encoder`. For the random frozen-feature control, pass `--freeze-encoder` without `--init-encoder`. To fine-tune, pass `--init-encoder` and omit `--freeze-encoder`. Match seeds, splits, and classifier training settings.

### FM-3 — Dense and cross-dataset transfer

Evaluate a spatial-label task (such as pet segmentation) with mIoU and boundary metrics and a separate fine-grained classification task. Compare scratch, PIXREL-pretrained, and appropriate published/pretrained reference models. Document preprocessing, license, and split provenance.

### FM-4 — Broader pretraining data

Only after the earlier gates show value, select a larger, diverse corpus whose license and permitted use are documented. Record dataset source/version, filtering, exclusions, checksums where feasible, and exact image counts. Add data scale gradually and report scaling curves; do not imply generality from CIFAR-100 alone.

## Run commands

Install the updated package after pulling the latest commit:

```bash
pip install -e .
```

First do an engineering-only check:

```bash
pixrel-pretrain --variant adaptive --seed 17 --epochs 1 --limit-train-batches 10 --batch-size 64 --device cuda --data ./data --output ./outputs/pretrain-smoke
```

Then run the pilot:

```bash
pixrel-pretrain --variant adaptive --seed 17 --epochs 100 --batch-size 128 --device cuda --data ./data --output ./outputs/pretrain-cifar100
```

Resume an interrupted pilot with the same configuration:

```bash
pixrel-pretrain --variant adaptive --seed 17 --epochs 100 --batch-size 128 --device cuda --data ./data --output ./outputs/pretrain-cifar100 --resume ./outputs/pretrain-cifar100/last.pt
```

The completed output includes `encoder.pt`, `last.pt`, `history.json`, and `pretrain.json`. Preserve all of them with the source commit and environment record.

## Stop/go criteria

Do not call the pilot a foundation model. Proceed to a larger corpus only if (a) the training is numerically stable without representational collapse, (b) frozen/finetuned results beat the same encoder from scratch on held-out data, and (c) at least one separate downstream task shows useful transfer. If adaptive PIXREL continues to trail a fair convolution control, revise the scientific question before spending more compute.
