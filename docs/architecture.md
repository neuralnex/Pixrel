# Architecture and tensor contracts

Input is a float tensor `B×3×H×W` (RGB normalized by the chosen dataset pipeline). PixelStem concatenates normalized x/y coordinates and maps five channels to 32 channels independently at each location.

The encoder returns `EncoderOutput` with:

- `embedding`: `B×64`, globally pooled from H3.
- `stages`: tuple H0–H3, with widths 32/32/48/64 and spatial sizes approximately H, H/2, H/4, H/8.
- `spatial`: fused dense features at H0 resolution by default, or H1 resolution when high-resolution fusion is disabled.
- `relationships`: optional `B×9×H×W` H0 neighborhood weights (adaptive/fixed variants only).

`classify=True` returns 100 class logits instead of the feature structure. The 3×3 relationship block shares its scoring/value/update functions across locations. Out-of-image neighbors are masked before softmax; the center is included. The fixed control uses uniform weights over valid neighbors. The convolution control replaces relationship blocks with residual 3×3 convolution blocks.

This is a first reference implementation. It prioritizes transparent semantics over speed; its per-neighbor tensor operations may be costly on CPU. Optimize only after profiling and preserve a reference-equivalence check when doing so.
