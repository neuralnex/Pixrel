# PIXREL research blueprint

## Mission

Study whether shared, context-adaptive local relationships between pixels, combined with hierarchical communication and access to high-resolution features, yield useful visual representations compared with controlled conventional baselines.

## Primary hypothesis

Adaptive local pixel relationships may improve representation quality over comparable convolution and fixed-aggregation controls. This remains unproven. Negative results are useful outcomes and must guide design decisions.

## Initial scope

Implement a transparent PixelStem, 3×3 relationship engine, four-stage hierarchy, multiscale feature fusion, global H3 embedding, classification head, and controlled variants. The CIFAR-100 V1 comparison is now recorded, but its fair-comparison gate remains open. A CIFAR-scale self-supervised pilot exists for engineering and representation analysis; do not scale it or add speculative architectural refinements before the next evidence gates.

## Gates

1. Engineering: tensor contracts, boundary masking, gradients, deterministic forward behavior.
2. Relationship benchmark: three controls × predetermined seeds on CIFAR-100. **Exploratory V1 results are recorded; fair-comparison gate remains open** because the convolution control has more parameters, the run is only 20 epochs, and compute matching has not been measured.
3. Dense-path engineering and ablation: a segmentation decoder now consumes and trains the fused spatial representation; synthetic CPU tests pass. Real mask-labeled evaluation and hierarchy/high-resolution-fusion ablations remain open.
4. Fine-detail tasks (for example CUB-200 and Oxford-IIIT Pet), after dataset licensing and protocol review.
5. Self-supervised pretraining pilot: CIFAR-100 and a small fire-image train split have completed masked local/fused/global target prediction with an EMA teacher and RGB reconstruction. CIFAR-100 pretrained frozen probes beat random frozen probes in-domain; the fire transfer comparison did not beat scratch.
6. Broader self-supervised pretraining and transfer remain gated on better controlled evidence, documented dataset rights, and successful evaluation on independent datasets/tasks.

Code existence is not evidence that a gate has passed. Current evidence does not establish broad transfer or foundation-model capability. The consolidated results, limitations, and release status are in the [final report](final_report.md). Checkpoints and full run archives are not included in this repository.

See [Relationship Value Benchmark V1](experiments/relationship-value-v1.md) for observed results, interpretation, documentation gaps, prior-art notes, and the ordered next steps.

See the [foundation-model research plan](foundation_model_plan.md) for the pilot objective, commands, evaluation gates, and stop/go criteria.
