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
3. Dense-path engineering and ablation: ensure the training objective uses the fused spatial representation, then evaluate hierarchy and high-resolution fusion.
4. Fine-detail tasks (for example CUB-200 and Oxford-IIIT Pet), after dataset licensing and protocol review.
5. Self-supervised pretraining pilot: code now exists for masked local/fused/global target prediction with an EMA teacher; it is unvalidated and CIFAR-100-scale only. Run diagnostics and linear probes before advancing.
6. Broader self-supervised pretraining and transfer, only after a defensible encoder, downstream evaluation protocol, and licensed data source exist.

Code existence is not evidence that a gate has passed. PIXREL has now been trained on CIFAR-100 for the exploratory runs recorded in the benchmark report, but those task-specific checkpoints are not a self-supervised pretrained encoder or a foundation model. The checkpoints and full run archives are not currently included in this repository.

See [Relationship Value Benchmark V1](experiments/relationship-value-v1.md) for observed results, interpretation, documentation gaps, prior-art notes, and the ordered next steps.

See the [foundation-model research plan](foundation_model_plan.md) for the pilot objective, commands, evaluation gates, and stop/go criteria.
