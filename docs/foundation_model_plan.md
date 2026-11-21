# PIXREL foundation-model research plan

## What the term means here

The target is a reusable visual encoder pretrained on diverse, permitted images without relying on downstream labels, then shown to transfer to multiple downstream tasks. The current CIFAR-100 supervised benchmark and the first CIFAR SSL probe are not a foundation model. The SSL probe did beat a random frozen encoder, but its 8.85% test accuracy is low and only demonstrates limited in-domain utility.

## Current pilot objective

The student receives an image with random patch-aligned regions replaced by zero in normalized input space. The EMA teacher receives the unmasked image. Student predictors learn normalized teacher targets at three levels:

1. H0 local pixel features, evaluated only at masked positions.
2. The fused high-resolution spatial feature map, evaluated only at masked positions.
3. The H3-pooled global image embedding.

In addition, RGB heads at H0–H3 predict the original normalized RGB values at masked positions. The total loss adds `rgb_weight` times the mean masked RGB smooth-L1 loss across those four heads. This makes pixel prediction directly trainable and visible, while the representation objectives remain in place. The teacher encoder is an EMA of the online encoder. Training logs per-objective losses, masked RGB MAE/PSNR, and feature-variance diagnostics, and writes encoder weights and resumable state.

This is a testable pilot design, not a claim of novelty or a validated pretraining recipe. Its choices should be compared with established approaches. Masked autoencoders reconstruct masked pixels with a visible-patch encoder and decoder ([He et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)); I-JEPA predicts target representations in embedding space ([Assran et al., 2023](https://arxiv.org/abs/2301.08243)); DINOv2 reports robust features from self-supervision with sufficiently large, curated, diverse data ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)). These are reference points, not methods PIXREL has matched or surpassed.

## Staged gates

### FM-0 — Fair encoder evidence

**Status: open.** The exploratory 20-epoch CIFAR comparison is complete, but the convolution baseline leads and has more parameters; compute matching is also missing. Complete a capacity-aware comparison before attributing behavior to the relationship operator. See [Relationship Value Benchmark V1](experiments/relationship-value-v1.md).

### FM-1 — Self-supervised engineering

**Status: engineering path complete for the current pilot.** CIFAR-100 and a fire-image training split have completed pretraining and produced diagnostics/checkpoints. The fire run's final masked RGB MAE was 0.06673 and PSNR was 20.20 dB. A decreasing loss or good reconstruction alone is not proof of useful features.

### FM-2 — Representation evaluation

Run frozen linear probes with the pretrained encoder and a same-architecture random-init control, using the same seed and data split. Then compare fine-tuning. Keep the official test split out of choices about architecture, schedule, and loss weights. CIFAR-100 results are in-domain representation evidence, not cross-domain transfer.

The CLI supports both probe controls. For the pretrained frozen probe, pass `--init-encoder <encoder.pt> --freeze-encoder`. For the random frozen-feature control, pass `--freeze-encoder` without `--init-encoder`. To fine-tune, pass `--init-encoder` and omit `--freeze-encoder`. Match seeds, splits, and classifier training settings.

**Status:** The three-seed CIFAR-100 frozen probe and official test evaluation are complete. Pretrained features scored 8.85% ± 0.06% test accuracy versus 3.29% ± 0.43% for random frozen features. This is a positive but weak within-dataset signal. Results and per-seed values are in [the experiment report](experiments/cifar100-ssl-seed17.md).

The three-seed fire transfer comparison is also complete. Pretraining averaged 94.08% test accuracy versus 95.83% from scratch, so the transfer-benefit gate did not pass on that dataset. Full metrics are in [Fire image transfer v1](experiments/fire-transfer-v1.md).

### FM-3 — Dense and cross-dataset transfer

**Status: not started.** No dense prediction or external dataset has been evaluated.

Evaluate a spatial-label task (such as pet segmentation) with mIoU and boundary metrics and a separate fine-grained classification task. Compare scratch, PIXREL-pretrained, and appropriate published/pretrained reference models. Document preprocessing, license, and split provenance.

### FM-4 — Broader pretraining data

**Status: gated.** The current fire dataset provenance and reuse rights are not documented, and the transfer result does not show an advantage. Do not scale the data or claim broad capability yet.

Only after the earlier gates show value, select a larger, diverse corpus whose license and permitted use are documented. Record dataset source/version, filtering, exclusions, checksums where feasible, and exact image counts. Add data scale gradually and report scaling curves; do not imply generality from CIFAR-100 alone.

## Current implementation milestone and result

The pretraining CLI now supports both CIFAR-100 and unlabeled `ImageFolder` directories, with configurable square resolution. ImageFolder labels are ignored, and images use the standard ImageNet normalization. Pretraining writes `encoder_bundle.pt` with weights, architecture variant, input size, and normalization metadata; `load_pretrained_encoder` exposes this checkpoint to Python callers. The `pixrel-transfer` and `pixrel-transfer-evaluate` commands support ImageFolder classification fine-tuning, frozen probes, scratch controls, and held-out evaluation. Training also records periodic input/mask, RGB reconstruction, stage-feature, fused-feature, local-neighbor-weight, loss, and masked RGB metric visualizations as PNGs and a GIF.

This is a reusable encoder and classification-transfer baseline, not broad foundation-model capability. The real ImageFolder smoke run and three-seed fire classification comparison are complete. The pretrained encoder averaged 94.08% test accuracy against 95.83% for scratch, so this experiment did not show a transfer benefit. No dense prediction task or external dataset has been evaluated. See the [final report](final_report.md) and [fire transfer experiment](experiments/fire-transfer-v1.md). Dataset provenance and permitted use must be recorded before broader training or redistribution.

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

The adaptive seed-17 CIFAR-100 self-supervised run and three-seed frozen-probe comparison, including the completed test evaluation, are recorded in [CIFAR-100 SSL seed 17](experiments/cifar100-ssl-seed17.md). Pretrained probes exceeded random frozen features but remained low in absolute accuracy. Together with the fire transfer result, this supports continuing controlled research, not scaling or making foundation-model claims. See the [final report](final_report.md) for the current evidence summary.
