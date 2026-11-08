# D=4 χ=128 GPU clean-environment sweep

Each row is a SEPARATE Julia invocation (no cross-arm contamination).

| precision | forloop | wall (s) | E_final | n_steps | ΔGPU (MB) |
|-----------|---------|----------|---------|---------|-----------|
| Float64 | 1 | 151.5 | -0.668966248003 | 20 | 14013 |
| Float32/coarse=2 | 1 | 147.2 | -0.668966913152 | 20 | 17536 |
| Float64 | 2 | 156.7 | -0.668966958148 | 20 | 19235 |
| Float32/coarse=2 | 2 | 153.1 | -0.668966924219 | 20 | 20373 |
| Float32/coarse=1 | 1 | 144.1 | -0.668964638779 | 20 | 14047 |
| wholeF32/coarse=2 | 1 | 157.3 | -0.668949710478 | 20 | 13678 |
