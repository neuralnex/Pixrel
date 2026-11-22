# PIXREL final report — current research snapshot

**Report date:** 2026-11-22  
**Scope:** CIFAR-100 relationship benchmark, CIFAR-100 self-supervised probe, and fire/non-fire ImageFolder pretraining/transfer runs reported from Colab. This is the final report for the current experimental stage, not a claim that the research program or model is complete.

**Implementation update:** a segmentation adapter and CLI have since been added to train the fused spatial pathway. Synthetic CPU tests pass; real mask-labeled evaluation remains outstanding. See [segmentation guide](segmentation.md).

**Validation status:** four CPU tests pass, including a synthetic one-epoch train/save/reload check. This validates code paths and tensor/checkpoint contracts only; no real segmentation dataset has been evaluated. Deployment remains gated on real-data validation and dataset-rights review.

**Annotation workflow:** a Colab guide now prepares a balanced 200-image subset from the existing fire split, creates all-background masks for the non-fire class, and interactively outlines visible flame regions in fire images. These human-reviewed masks must be completed and checked before the real segmentation run; none have been supplied yet.

## Executive summary

PIXREL is an experimental, pixel-relationship visual encoder with self-supervised masked feature prediction and masked RGB reconstruction. The current evidence shows limited in-domain representation utility on CIFAR-100 frozen probes. On the small fire-image experiment, however, the pretrained encoder did **not** improve held-out classification over the same architecture trained from scratch: mean accuracy was 94.08% pretrained versus 95.83% scratch across three seeds. The project has not established broad transfer, a segmentation capability, or foundation-model status.

The pretrained fire encoder can be shared as a research artifact with its limitations and results. The Hub upload tooling and model card are prepared for `nexusbert/pixrel-fire-encoder-v1`; actual publication, dataset rights verification, and a model license remain outstanding.

## Current model and pixel-level objective

The adaptive encoder uses a pixel-and-coordinate stem, four hierarchical stages, shared 3×3 content-adaptive neighbor aggregation, a global embedding, and fused spatial features. The SSL trainer masks patch-aligned input regions, predicts EMA-teacher features at local, fused-spatial, and global levels, and predicts masked RGB values from encoder stages H0–H3. Training visualizations include masked input, reconstruction predictions/errors, normalized feature responses, and local neighbor weights. These maps are diagnostics; they do not establish causal paths.

## Fire-image pretraining

- Data path in Colab: `Fire_detection_model/fire_dataset`; the benchmark split was created as 70% train, 15% validation, and 15% test per class.
- Pretraining used only the training split, so validation and test images were excluded from the unlabeled pretraining run.
- Adaptive variant, seed 17, 128×128, 20 epochs, patch size 4, batch size 8, RGB weight 0.1, NVIDIA L4.
- Final epoch: total loss 0.05651; local pixel cosine loss 0.04255; fused spatial cosine loss 0.00156; global cosine loss 0.000486; masked RGB loss 0.12155; masked RGB MAE 0.06673 on [0,1] RGB; masked PSNR 20.20 dB.

These metrics show that the configured objective trained and the model predicted masked pixels. They do not measure semantic quality or task utility.

## Fire classification transfer

Both models used the same adaptive architecture, train/validation/test split, 30 fine-tuning epochs, batch size 32, and NVIDIA L4. Pretrained weights were used for one condition; scratch initialization was the control. Test results were supplied from Colab:

| Seed | Pretrained loss | Pretrained accuracy | Scratch loss | Scratch accuracy |
|---:|---:|---:|---:|---:|
| 17 | 0.294014 | 94.74% (144/152) | 0.185291 | 94.74% (144/152) |
| 29 | 0.355838 | 93.42% (142/152) | 0.185809 | 96.05% (146/152) |
| 43 | 0.295260 | 94.08% (143/152) | 0.204674 | 96.71% (147/152) |
| Mean across seeds | 0.315038 | **94.08%** | 0.191924 | **95.83%** |

Pretraining trails by 1.75 percentage points in mean accuracy. Seed 17’s best validation accuracy was 97.99% pretrained and 97.32% scratch, a one-image difference on the small validation split; this did not predict the mean test result. The same 152 test images were evaluated for each seed, so the 456 run-image observations are not 456 independent images. No class-wise metrics, confusion matrices, confidence intervals, or external dataset evaluation were supplied.

## Other completed evidence

- CIFAR-100 adaptive/fixed/conv supervised benchmark: adaptive averaged 43.22% test accuracy, fixed 41.94%, and convolution 50.52% over three seeds. Convolution had more parameters, and the comparison was only 20 epochs; see [Relationship Value Benchmark V1](experiments/relationship-value-v1.md).
- CIFAR-100 SSL frozen probes: pretrained features averaged 8.85% test accuracy versus 3.29% for random frozen features over three seeds. This is limited in-domain linear utility on a 100-class dataset; see [CIFAR-100 SSL pilot](experiments/cifar100-ssl-seed17.md).

## Conclusions and boundaries

Supported: the code trains the intended pixel/feature objectives; the CIFAR-100 pretrained encoder has limited linear utility over random frozen features; the current fire transfer experiment does not show a gain over scratch.

Not supported: superiority over convolution, broad cross-domain transfer, reliable fire detection, causal pixel paths, or foundation-model status. The fire dataset’s source/version and reuse terms were not documented. The reported test set is small and only one dataset/task was evaluated.

## Release and next research work

The model card and upload workflow are in [the Hugging Face release guide](huggingface/publish.md). Upload is not yet confirmed. Verify dataset and derived-weight rights, then select a license before claiming that others may redistribute or reuse the weights.

Prioritize documenting dataset provenance and class balance; add class-wise metrics and uncertainty; check duplicates and acquisition-level leakage; reproduce on an external fire dataset; and evaluate the new segmentation adapter on a permitted mask-labeled task. Continue comparing pretrained against scratch and relevant matched baselines. Treat the current Hub checkpoint as an experimental contribution point only.
