# Relationship Value Benchmark V1

## Question

Under one fixed CIFAR-100 training protocol, do adaptive pixel relationship blocks outperform a convolutional control or uniform local aggregation control?

## Conditions

Run `adaptive`, `fixed`, and `conv`, with seeds 17, 29, and 43. Keep input resolution, splits, augmentations, optimizer, schedule, batch size, epochs, and evaluation rules identical. Each run creates a deterministic 45,000/5,000 train/validation split from the official training split. Select checkpoints by validation accuracy and evaluate the official test split once per selected checkpoint.

## Record

Retain each command/config, seed, source revision, dependency versions, train/validation history, best checkpoint, test metrics, parameter count, elapsed time, CPU thread count, and peak resident memory if measured. Report per-seed results and mean ± standard deviation. Do not treat limited-batch/short-epoch runs as benchmark evidence. Parameter counts are not matched by this initial implementation, so report them and avoid attributing differences solely to the relationship mechanism if capacity differs materially.

## CPU execution

The scripts place tensors and models on CPU and never call CUDA APIs. Use `--threads` to set PyTorch CPU intra-op threads; `--workers 0` is the reproducible default. Training can take a long time on a small CPU. First check the pipeline with a limited run, then launch full conditions when runtime permits.

## Current limitations

This initial benchmark is classification only. It does not establish fine-detail preservation, transfer, pretraining quality, or foundation-model status. The controls are not yet matched for MACs. CIFAR-100 is a first controlled benchmark, not a sufficient generalization study.
