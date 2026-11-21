# Relationship Value Benchmark V1 — CIFAR-100

This CIFAR benchmark is one component of the current evidence snapshot. The later fire-image transfer experiment and consolidated conclusions are in the [final report](../final_report.md).

## Status

**Exploratory three-seed run completed; primary hypothesis not established.** The adaptive relationship encoder exceeded uniform fixed aggregation, but the convolution control scored higher. Capacity and training-duration confounds mean this is not yet a fair claim about the relationship operator itself.

## Research question

Under a shared PIXREL stem and hierarchy, does content-adaptive local relationship aggregation improve CIFAR-100 image classification over uniform neighborhood aggregation and a convolutional block?

## Run protocol recorded from the commands and result output

- Dataset: CIFAR-100, 32×32 RGB, 100 classes; official training split divided into 45,000 train and 5,000 validation examples; official 10,000-example test split used for the reported final metrics.
- Conditions: `adaptive`, `fixed`, and `conv`.
- Seeds: 17, 29, and 43 for every condition.
- Training command settings: 20 epochs, batch size 64, CUDA, NVIDIA L4. The training script defaults to SGD, learning rate 0.1, momentum 0.9, weight decay 0.0005, cosine annealing over the requested epoch count, random crop with padding 4, random horizontal flip, and CIFAR-100 channel normalization.
- Model selection: best checkpoint selected using validation accuracy; test evaluation then run on that checkpoint.
- Metrics below were copied from the user-provided Colab evaluation output. Means and sample standard deviations were recomputed from those nine results. Standard deviations are descriptive across three seeds, not confidence intervals.
- The training command seeds Python, NumPy, and PyTorch, but deterministic CUDA kernels are not explicitly enabled. Treat seeds as controlled random initializations, not a guarantee of bitwise-reproducible GPU runs.

Exact source commit, dependency versions, per-epoch histories, elapsed time, peak memory, and per-run training summaries were not included with the pasted evaluation output. Preserve `run.json`, `history.json`, `eval.json`, the exact command, and `git rev-parse HEAD` from each Colab run to complete the record. Do not reconstruct missing values from memory.

## Test results

Accuracy is top-1 percentage. The displayed loss is the script's arithmetic mean of per-batch mean cross-entropies. Since the 10,000-example test set does not divide evenly into batches of 64, its final partial batch receives the same weight as a full batch; the loss should be recomputed with example-weighted aggregation before treating small loss differences as definitive. Mean ± SD uses sample standard deviation (`n=3`).

| Variant | Accuracy by seed 17 / 29 / 43 (%) | Accuracy mean ± SD (%) | Loss by seed 17 / 29 / 43 | Loss mean ± SD |
|---|---:|---:|---:|---:|
| Adaptive | 42.69 / 43.40 / 43.56 | 43.22 ± 0.46 | 2.1842 / 2.1630 / 2.1665 | 2.1712 ± 0.0113 |
| Fixed aggregation | 42.13 / 41.87 / 41.81 | 41.94 ± 0.17 | 2.2153 / 2.2479 / 2.2262 | 2.2298 ± 0.0166 |
| Convolution | 50.61 / 49.74 / 51.22 | 50.52 ± 0.74 | 1.8379 / 1.8450 / 1.8124 | 1.8318 ± 0.0171 |

### Interpretation

- Adaptive exceeded fixed aggregation by **1.28 percentage points** in mean accuracy and had lower mean loss by **0.0586**. This is preliminary evidence that learned content-dependent weights help relative to uniform aggregation in this specific implementation and training setup.
- Convolution exceeded adaptive by **7.31 percentage points** and had lower mean loss by **0.3394**. The current results do **not** support the claim that PIXREL's adaptive relationship encoder is better than the convolution control on CIFAR-100.
- The convolution model has **192,020** parameters in the checked-in configuration; adaptive and fixed each have **150,488** counted parameters. Thus convolution has about **27.6%** more parameters than adaptive. These counts were computed from the current model code, not recovered from the Colab `run.json` files.
- Fixed mode still instantiates the adaptive score network, although its forward pass does not use it. Its raw parameter count therefore overstates active parameters and makes parameter-count comparisons alone incomplete.
- No MAC/FLOP, throughput, latency, memory, energy, or parameter-matched results are available. Three seeds give a useful pilot summary but are not enough to establish broad superiority or generalization.

## Important scope limit in the current implementation

The classification path returns `classifier(H3 pooled)`. It does not use `EncoderOutput.spatial`, the fused multiscale feature map. The fusion branch is computed in `forward`, but CIFAR classification loss does not train its weights. Therefore this experiment evaluates the H3 global embedding and classification head; it does not validate dense features, high-resolution fusion, small-object preservation, segmentation, or boundary quality. This should be corrected or explicitly isolated before using classification results as evidence for the dense/foundation-model path.

## Research decision

Do not scale to large self-supervised pretraining yet. First establish a capacity-aware, adequately trained comparison and determine whether the local relationship operator contributes beyond a matched convolutional alternative. A published CIFAR-100 protocol has, for example, trained CNN baselines for 300 epochs; that is not a universal requirement, but it makes this 20-epoch result an early-budget comparison rather than a mature convergence claim ([WACV 2024 supplemental baseline protocol](https://openaccess.thecvf.com/content/WACV2024/supplemental/Meronen_Fixing_Overconfidence_in_WACV_2024_supplemental.pdf)). Keep epochs and schedule identical across variants and report them explicitly.

## Next research steps

1. **Complete the run archive.** Copy each variant/seed's `run.json`, `history.json`, and `eval.json` from Drive. Record repository commit, package versions, commands, and hashes for checkpoints and the dataset archive where practical.
2. **Audit capacity and compute.** Create an actually parameter-matched convolution control. Remove or freeze the unused score network in the fixed control, and report active/total parameters, MACs, peak memory, and throughput.
3. **Run a stronger controlled comparison.** Choose a longer, fixed schedule using validation only. Train all three variants with the same epochs, augmentation, batch size, optimizer, schedule, and seeds. Do not use test results to tune the new runs. Evaluate the selected checkpoints once on test after the protocol is frozen. Compute loss weighted by example count, including the final partial batch.
4. **Inspect whether adaptive weights are meaningful.** Record relationship-weight entropy, neighbor concentration, center/self-weight, and examples of maps on correct and incorrect predictions. Compare against the fixed control; visualization alone is not evidence of causal importance.
5. **Test the actual fine-detail claim.** Add a dense head and a dataset/task with spatial labels (for example Oxford-IIIT Pet segmentation), then measure mIoU and boundary quality. The current classifier does not train the fused spatial pathway.
6. **Review prior art before a novelty claim.** Local Relation Networks already describe adaptive aggregation based on relationships between local pixel pairs ([Hu et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Hu_Local_Relation_Networks_for_Image_Recognition_ICCV_2019_paper.html)). Dense pixel-level self-supervised representations also predate this project; VADeR is one example ([Pinheiro et al., 2020](https://arxiv.org/abs/2011.05499)). PIXREL needs a clearly defined, experimentally supported distinction beyond “relations begin at pixels.”
7. **Only then begin small-scale pretraining.** Use a documented, license-reviewed real-image corpus and a stable self-supervised objective. Evaluate frozen linear probes and fine-tuning against the same encoder from scratch on multiple downstream tasks. Call it a small pretrained research encoder unless transfer breadth and evidence justify a stronger label.

## Claims supported today

Supported: in this 20-epoch CIFAR-100 experiment, adaptive PIXREL scored above its fixed-aggregation variant; the checked-in convolution variant scored above adaptive PIXREL; all figures are based on three seeds.

Not supported: PIXREL is better than convolution for vision, preserves fine details, transfers broadly, is novel, or is a foundation model.
