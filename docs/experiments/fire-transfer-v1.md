# Fire image transfer v1

## Summary

This Colab experiment compared an adaptive PIXREL encoder pretrained on the training portion of a fire/non-fire image dataset with the same encoder architecture initialized from scratch. Both were fine-tuned with seed 17, 29, and 43, for 30 epochs, batch size 32, and NVIDIA L4. The held-out test split contained 152 images. The reported results were pasted by the project owner from Colab.

The scratch baseline had higher test accuracy for seeds 29 and 43 and tied at seed 17. Across seeds, average accuracy was 95.83% for scratch and 94.08% for pretrained, a difference of -1.75 percentage points for pretraining. This experiment does not show an improvement from pretraining.

## Test results

| Seed | Pretrained loss | Pretrained accuracy | Scratch loss | Scratch accuracy |
|---:|---:|---:|---:|---:|
| 17 | 0.294014 | 94.74% | 0.185291 | 94.74% |
| 29 | 0.355838 | 93.42% | 0.185809 | 96.05% |
| 43 | 0.295260 | 94.08% | 0.204674 | 96.71% |
| Mean | 0.315038 | 94.08% | 0.191924 | 95.83% |

Each seed was evaluated on the same 152-image test split. Treating the three runs as repeated model fits, the arithmetic mean of their accuracies is 94.08% vs 95.83%. The combined counts across seed evaluations are 429/456 and 437/456 respectively; this is not 456 independent test examples because images are reused across seeds.

## Validation results

For seed 17, best validation accuracy was 97.99% for pretrained and 97.32% for scratch. The validation split was small, and that difference corresponds to one image. Validation results do not reverse the test comparison.

## Pretraining setup

- Adaptive PIXREL encoder; 128×128 inputs; patch size 4.
- 20 unlabeled pretraining epochs, batch size 8, RGB reconstruction weight 0.1.
- Pretraining was run only on the training split to keep validation/test images out of the unlabeled pretraining set.
- Fine-tuning used 30 epochs, batch size 32, learning rate 0.001, cosine schedule, and seeds 17, 29, and 43.
- Test split size: 152 images.

## Limitations and next research steps

- Dataset identity/version, collection procedure, class balance, and licensing were not provided in the Colab output. Record these before publication or redistribution.
- One small test split and three seeds do not establish broad generalization. The shared test images mean seed runs are paired, not independent samples.
- No class-wise precision/recall, confusion matrix, confidence intervals, calibration, or external dataset results were reported.
- The scratch baseline performed better on average. Investigate transfer protocol, pretraining scale/objective, and dataset characteristics before making any benefit claim.
- Fire detection can be safety-critical. This model is a research prototype and must not be presented as a validated detection or alarm system.
