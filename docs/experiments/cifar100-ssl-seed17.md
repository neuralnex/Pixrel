# CIFAR-100 self-supervised pilot — adaptive seed 17

**Final status:** the planned official test evaluation is complete. This remains a narrow in-domain probe result, not evidence of broad transfer or foundation-model status. See the [consolidated final report](../final_report.md).

## Run record

- Status: completed successfully, based on the pasted Colab log.
- Data: 45,000 unlabeled images from the CIFAR-100 training split; the validation and official test images were not included in pretraining.
- Hardware: NVIDIA L4.
- Configuration: adaptive encoder, seed 17, 100 epochs, batch size 128, 352 batches per epoch, learning rate 0.0003, weight decay 0.05, 4x4 masking at ratio 0.4, global loss weight 0.5, spatial loss weight 1.0, EMA momentum 0.99 to 0.999.
- Elapsed time: 5,182.4 seconds (about 1 hour 26 minutes).
- Encoder parameters: 150,488.
- Checkpoint directory reported by the run: `/content/drive/MyDrive/pixrel-runs/pretrain-cifar100`.
- The run summary says the checkpoint is a CIFAR-scale SSL pilot, not a foundation-model result.

## Selected training diagnostics

| Epoch | Total loss | Local pixel | Fused spatial | Global | Teacher feature std | Teacher embedding channel std |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2806 | 0.2170 | 0.0509 | 0.0253 | 0.4904 | 0.0057 |
| 9 | 0.0346 | 0.0332 | 0.0006 | 0.0015 | 0.5301 | 0.1524 |
| 20 | 0.1039 | 0.0877 | 0.0083 | 0.0157 | 0.4076 | 0.4312 |
| 50 | 0.1506 | 0.1284 | 0.0143 | 0.0158 | 0.3550 | 0.4505 |
| 100 | 0.1312 | 0.1189 | 0.0074 | 0.0097 | 0.3301 | 0.5266 |

The total loss reached its minimum at epoch 9, then increased and settled around 0.13. This is not evidence that the representation is useful, and it warrants inspecting the complete curves and the saved encoder before changing the training recipe. The logged feature standard deviations do not show all features becoming constant, but they are only coarse diagnostics and cannot rule out partial collapse or poor representations. EMA cosine losses can move as the teacher changes, so loss increase alone does not prove failure either.

## Interpretation and next gate

At this stage of the experiment, the 100-epoch pretraining job had completed but transfer quality was not yet measured. The planned matched frozen linear-probe comparison was:

1. Pretrained adaptive encoder, frozen; initialize from `encoder.pt`.
2. Randomly initialized adaptive encoder, frozen; omit `--init-encoder`.

Run both with seeds 17, 29, and 43, preserve validation selection, and evaluate the chosen checkpoints once on the official CIFAR-100 test split. Report each seed and mean ± standard deviation. If pretrained frozen features do not outperform random frozen features, do not scale the corpus yet; inspect the learning curves, normalization, predictor/target dynamics, and objective weights first. If they do, follow with a matched fine-tuning comparison, then a separate dense prediction dataset before considering a larger corpus.

The exact commands are in [the README](../../README.md#self-supervised-learning-pilot) and the decision gates are in [the foundation-model plan](../foundation_model_plan.md). The later probe and test results below update the initial next-step status. These results are on one small dataset and must not be described as a foundation model or a breakthrough.

## Frozen linear-probe follow-up

The three-seed frozen probes were completed on an NVIDIA L4 with the same per-seed split and classifier settings for pretrained and random encoders. The best validation accuracies from each `run.json` were:

| Seed | Pretrained frozen | Random frozen | Difference |
|---:|---:|---:|---:|
| 17 | 8.64% | 3.16% | +5.48 pp |
| 29 | 9.00% | 3.70% | +5.30 pp |
| 43 | 9.40% | 3.56% | +5.84 pp |
| Mean ± sample SD | 9.01% ± 0.38 pp | 3.47% ± 0.28 pp | +5.54 pp |

At validation time, pretrained features were more linearly useful than this encoder's random features on CIFAR-100. However, 9.01% top-1 validation accuracy remains low for a 100-class task, and this comparison alone does not show competitive representations or transfer beyond CIFAR-100. Treat it as an early positive signal, not a foundation-model result. The official test evaluation is now recorded below. Do not change this recipe based on the already-used test results.

## Official test evaluation

The six selected checkpoints were evaluated on the official CIFAR-100 test split:

| Seed | Pretrained frozen | Random frozen | Difference |
|---:|---:|---:|---:|
| 17 | 8.84% | 3.05% | +5.79 pp |
| 29 | 8.91% | 3.78% | +5.13 pp |
| 43 | 8.80% | 3.04% | +5.76 pp |
| Mean ± sample SD | 8.85% ± 0.06 pp | 3.29% ± 0.43 pp | +5.56 pp |

The pretrained checkpoints outperform their random-feature controls on the held-out test set for all three seeds, closely matching the validation direction. This supports a narrow conclusion: the current CIFAR-100 SSL run learned features with some in-domain linear utility. The absolute accuracy remains low. The next research step is a controlled end-to-end fine-tuning comparison, followed by a separate dataset/task; do not tune on this already-used test set.
