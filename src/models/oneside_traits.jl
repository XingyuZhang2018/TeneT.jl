"""
    _oneside_down_index(::Type{<:Model}, i, Ni) -> Int

Row index `ir` such that `ACd[ir, j] ≡ ACu[i, j]` under the model's up-down
hermiticity, when using `VUMPS{<:Oneside}`. Dispatched on the model type
(not instance) because `Oneside{M}` carries only the type at the algorithm level.

# Default
Returns `Ni + 1 - i` — reflection across the unit cell middle. Standard for
models with sublattice-level U-D hermiticity.

# Overrides
Per-model overrides go in the model's `order_init.jl`. For example,
`J1J2p{Honeycomb{:brickwall_v}}` under single-site restriction uses
`ir = i` because each iPEPS tensor is u-d self-symmetric.
"""
_oneside_down_index(::Type{<:HamiltonianModel}, i, Ni) = Ni + 1 - i
