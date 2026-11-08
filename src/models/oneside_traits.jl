"""
    obs_index(::Type{<:Model}, i, Ni) -> Int

Row index `ir` for observation environments (`ifobs=true`) such that
`ACd[ir, j] ≡ ACu[i, j]` under the model's up-down hermiticity. Dispatched
on the model type (not instance) and threaded through `leftenv` / `rightenv`
via the `model` keyword.

# Default
Returns `Ni + 1 - i` — reflection across the unit cell middle. Standard for
models with sublattice-level U-D hermiticity.

# Overrides
Per-model overrides go in the model's `order_init.jl`. For example,
`J1J2p{Honeycomb{:brickwall_v}}` under single-site restriction uses
`ir = i` because each iPEPS tensor is u-d self-symmetric.
"""
obs_index(::Type{<:HamiltonianModel}, i, Ni) = Ni + 1 - i

"""
    uses_oneside_obs_env(::Type{<:Model}) -> Bool

Whether a single-runtime `ObsEnv` should return `OnesideVUMPSEnv` instead of
the legacy `VUMPSEnv` shape. This is separate from `obs_index`: ordinary
models may still pass through `model` to select the default reflected row
without opting into the memory-saving one-sided environment layout.
"""
uses_oneside_obs_env(::Type{<:HamiltonianModel}) = false
