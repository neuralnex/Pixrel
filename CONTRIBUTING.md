# Contributing to PIXREL

PIXREL is an experimental computer vision research project. Contributions that test, reproduce, or improve the encoder are welcome.

Read the [current final research report](docs/final_report.md) before proposing claims or uploading a derived checkpoint; the current fire transfer results do not show a pretrained advantage.

## Model Hub contributions

- Open an issue or pull request on the model repository with the task, dataset source/version, split method, preprocessing, training command, checkpoint provenance, and evaluation results.
- Compare against an identical PIXREL architecture trained from scratch. Include multiple seeds when practical, and select checkpoints using validation data only.
- Report held-out test results once after model selection. Include per-class precision/recall or a confusion matrix for imbalanced classification tasks.
- Do not upload images, labels, private data, credentials, or derived weights unless their terms permit the upload and redistribution.
- Do not describe normalized feature maps as causal explanations or describe the current project as a validated foundation model.

## Code contributions

Keep dataset-specific code separate from the core encoder where possible. Document CLI changes and record the exact environment, seed, and command for research changes. Do not commit generated datasets, checkpoints, or Hub tokens.

## Licensing

The repository does not currently declare a software or model license. Until maintainers add one after checking relevant rights, contributions and artifacts should not be assumed to carry a particular reuse license.
