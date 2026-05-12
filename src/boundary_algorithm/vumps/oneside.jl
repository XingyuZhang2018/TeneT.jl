# ============================================================================
# Oneside VUMPS — for models with up-down (but NOT left-right) hermiticity
#
# Key differences from General VUMPS:
#   - Only ONE VUMPS iteration (no separate down VUMPS); down env derived via
#     model trait `_oneside_down_index(::Type{<:Model}, i, Ni)`
#   - ObsEnv builds FLo/FRo via leftenv_oneside / rightenv_oneside (which
#     use the model trait, not hardcoded `ir = Ni+1-i` of `ifobs=true`)
#   - Reuses VUMPSRuntime (5 fields); introduces OnesideVUMPSEnv (6 fields)
#
# Model opt-in: implicit via existence of `energy_value(::Model, A,
# env::OnesideVUMPSEnv, params)`. No explicit trait.
# ============================================================================

# Note: vumps_step, vumps_itr, leftenv_oneside, rightenv_oneside, ObsEnv,
# imag_error all live in this file (added in subsequent Stage 2 tasks).

# ── Initialization ──────────────────────────────────────────────────

"""
    init_env(M, χ, alg::VUMPS{<:Oneside})

Initialize a single VUMPSRuntime for Oneside mode. No down VUMPS is created
since the down env is derived from the up env via `_oneside_down_index` at
ObsEnv time.
"""
function init_env(M::StructArray, χ::Int, alg::VUMPS{<:Oneside})
    alg.ifparallelupdown && error("Oneside is incompatible with ifparallelupdown=true")
    if alg.ifupdown
        alg.verbosity >= 2 && @info "Oneside ignores ifupdown=true; running single-side VUMPS only"
    end
    Ni, Nj = size(M)
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    C = LRtoC(L, R)
    if alg.ifparallel
        AL = MPI.bcast(AL, 0, MPI.COMM_WORLD)
        AR = MPI.bcast(AR, 0, MPI.COMM_WORLD)
        C = MPI.bcast(C, 0, MPI.COMM_WORLD)
    end
    _, FL = leftenv(AL, conj(AL), M; alg)
    _, FR = rightenv(AR, conj(AR), M; alg)
    alg.verbosity >= 2 && ChainRulesCore.ignore_derivatives(() ->
        @info "Oneside VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ)")
    return VUMPSRuntime(AL, AR, C, FL, FR)
end

# ── leading_boundary entry ───────────────────────────────────────────

"""
    leading_boundary(rt::VUMPSRuntime, M, alg::VUMPS{<:Oneside})

Run the Oneside VUMPS iteration. Returns the converged runtime and the
final error. Note: `vumps_itr` for Oneside is added in Task 6 — this entry
will MethodError until then. Stage 2 Task 6 lands the iteration body.
"""
function leading_boundary(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{<:Oneside})
    return vumps_itr(rt, M, alg)
end
