# PIXREL research blueprint

## Mission

Study whether shared, context-adaptive local relationships between pixels, combined with hierarchical communication and access to high-resolution features, yield useful visual representations compared with controlled conventional baselines.

## Primary hypothesis

Adaptive local pixel relationships may improve representation quality over comparable convolution and fixed-aggregation controls. This remains unproven. Negative results are useful outcomes and must guide design decisions.

## Initial scope

Implement a transparent PixelStem, 3×3 relationship engine, four-stage hierarchy, multiscale feature fusion, global H3 embedding, classification head, and three benchmark variants. Establish CIFAR-100 training/evaluation and reproducibility before adding self-supervised pretraining or evidence-preserving refinements.

## Gates

1. Engineering: tensor contracts, boundary masking, gradients, deterministic forward behavior.
2. Relationship benchmark: three controls × predetermined seeds on CIFAR-100.
3. Ablation: hierarchy and high-resolution fusion.
4. Real-data self-supervised pretraining and transfer.
5. Fine-detail tasks (for example CUB-200 and Oxford-IIIT Pet), after dataset licensing and protocol review.

Code existence is not evidence that a gate has passed. The current implementation is untrained and is not a pretrained or foundation model.
