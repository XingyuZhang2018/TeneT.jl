# ============================================================================
# Honeycomb{:brickwall_pi6} — alternative brickwall embedding for honeycomb
# ============================================================================
#
# This module implements an alternative embedding of the honeycomb lattice
# into the square iPEPS lattice. Compared with `Honeycomb{:brickwall}`, the
# honeycomb is rotated by π/6 (equivalently, the brickwall offset runs along
# columns instead of rows). The bond on each site that is "removed" by setting
# its dimension to 1 is HORIZONTAL (L or R), not vertical (U or D).
#
#                              :brickwall                 :brickwall_pi6
#   leg conventions       L=D, D=1, R=D, U=D           L=1, D=D, R=D, U=D   (even-parity)
#                         L=D, D=D, R=D, U=1           L=D, D=D, R=1, U=D   (odd-parity)
#
# Consequence: every iPEPS row has BOTH non-trivial U and D bonds, so the
# transfer matrix going up is identical to the transfer matrix going down.
# Up and down VUMPS environments are therefore related by mirror reflection,
# and you can set `ifdownfromup=true` to save half the boundary work.
#
# This module defines:
#   * `_init_random_ipeps(::Honeycomb{:brickwall_pi6}, ...)`  — tensor shape
#   * `_lattice_map(A, ::Honeycomb{:brickwall_pi6}, pattern)` — odd-parity permute
#   * `enlarge_coupling(model::J1J2p{Honeycomb{:brickwall_pi6}}, ...)`
#   * `energy_value(model::J1J2p{Honeycomb{:brickwall_pi6}}, A, env, params)`
#
# To use a non-uniform Nj×Ni pattern such as Ni=6, Nj=2 (analog of the original
# Ni=2, Nj=6 brickwall pattern), call `J1J2p(lattice=Honeycomb(:brickwall_pi6), …)`.

# ----------------------------------------------------------------------------
#  Initialisation: tensor shape (1, D, D, D, d, N)
# ----------------------------------------------------------------------------

function _init_random_ipeps(::Honeycomb{:brickwall_pi6}, etype, D, d, N, Ni, Nj)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError(
        "For Honeycomb{:brickwall_pi6}, both Ni and Nj must be even."))
    # Leg order is (L, D, R, U, phys). L is the trivial (dim-1) leg.
    rand(etype, 1, D, D, D, d, N) .+ 1
end

# ----------------------------------------------------------------------------
#  Hermitian-VUMPS symmetrisation: enforce A[l,d,r,u,p] = A[l,u,r,d,p]
# ----------------------------------------------------------------------------
#
# When the raw iPEPS tensor is invariant under swapping the D (index 2) and U
# (index 4) virtual legs, every M-tensor `M = A * conj(A)` inherits the same
# D↔U mirror symmetry, the column transfer matrix becomes hermitian under that
# mirror, and the up/down VUMPS environments are exactly related by reflection
# (so `ifdownfromup=true` is exact, not approximate).
#
# The lattice map for `:brickwall_pi6` only permutes legs 1↔3 (L↔R) on
# odd-parity sites — it never touches legs 2 (D) and 4 (U) — so this single
# constraint on the raw `A` carries through to the post-mapped iPEPS uniformly.
#
# `dumu_symmetrize(A)` projects an iPEPS parameter array onto the symmetric
# subspace. Use it both at initialisation and as the `restriction_ipeps` hook
# during gradient optimisation.

"""
    dumu_symmetrize(A)

Symmetrise an iPEPS parameter array under the D↔U leg swap:

    A_sym[l, d, r, u, p, i] = (A[l, d, r, u, p, i] + A[l, u, r, d, p, i]) / 2

Works on rank-6 arrays (the format used by `init_ipeps`). The result is an
exact fixed point of the swap, i.e.
`permutedims(A_sym, (1, 4, 3, 2, 5, 6)) ≈ A_sym`.
"""
function dumu_symmetrize(A::AbstractArray{T, 6}) where T
    return (A .+ permutedims(A, (1, 4, 3, 2, 5, 6))) ./ 2
end

"""
    init_ipeps_pi6_hermitian(; atype, etype, No, D, χ, params)

Convenience wrapper around `init_ipeps` that immediately applies
`dumu_symmetrize` to the random initial state. Use together with
`restriction_ipeps = dumu_symmetrize` and `ifdownfromup = true` in the
VUMPS algorithm settings.
"""
function init_ipeps_pi6_hermitian(; atype=Array, etype=Float64, No::Int=0,
                                    D::Int, χ::Int, params::iPEPSOptimize)
    A = init_ipeps(; atype, etype, No, D, χ, params)
    return atype(dumu_symmetrize(Array(A)))
end

# ----------------------------------------------------------------------------
#  Lattice map: permute (3, 2, 1, 4, 5) on odd-parity sites
# ----------------------------------------------------------------------------
#
#  Even parity (i+j even):  legs stay  (L=1, D=D, R=D, U=D)
#  Odd parity  (i+j odd):   legs become (L=D, D=D, R=1, U=D)
#  i.e. for odd sites we swap legs 1 (L) ↔ 3 (R) while keeping D, U, phys.

function _lattice_map(A, ::Honeycomb{:brickwall_pi6}, pattern)
    Ni, Nj = size(pattern)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError(
        "For Honeycomb{:brickwall_pi6}, pattern must have even Ni and Nj."))
    n_unique = length(unique(pattern))

    # Each unique tensor must occupy positions of a single parity (i+j mod 2).
    # Otherwise the same tensor would land at both an L-trivial slot (even) and
    # an R-trivial slot (odd) without the necessary L↔R permute, which silently
    # produces a mis-shaped FL/FR boundary tensor downstream.
    for i in 1:n_unique
        positions = findall(==(i), pattern)
        parities  = unique([sum(Tuple(p)) % 2 for p in positions])
        length(parities) == 1 || throw(ArgumentError(
            "Honeycomb{:brickwall_pi6} requires each unique tensor to appear " *
            "only at positions with the same (i+j) % 2. Tensor $i appears at " *
            "$(Tuple.(positions)), which has mixed parities."))
    end

    return StructArray([
        begin
            pos = findfirst(==(i), pattern)
            sum(Tuple(pos)) % 2 == 0 ? A[i] : permutedims(A[i], (3, 2, 1, 4, 5))
        end
        for i in 1:n_unique
    ], pattern)
end

# ----------------------------------------------------------------------------
#  Coupling enlargement for J1J2p on pi6 (uniform and plaquette)
# ----------------------------------------------------------------------------

function enlarge_coupling(model::J1J2p{Honeycomb{:brickwall_pi6}}, ::Val{:uniform}, i, j)
    J1 = model.J1
    return J1, J1   # (J1h, J1v)
end

# Plaquette pattern for the Nj=2 / Ni-even patterns. By convention we let
# `bondratio` weaken either the J1H or J1V bond on a specific subset of sites.
# This mirrors the structure of the original brickwall plaquette enlargement
# but with rows ↔ columns swapped.  Customise as needed for the pattern in use.
function enlarge_coupling(model::J1J2p{Honeycomb{:brickwall_pi6}}, ::Val{:plaquette}, i, j)
    @unpack J1, bondratio = model
    J1h = J1v = J1
    # 90° rotated version of the original plaquette mask (which acted on
    # the 2×6 pattern [1 3 5 2 4 6; 2 4 6 1 3 5]). For the conjugate 6×2
    # pattern, the J1H/J1V weakened sites swap (rows ↔ cols).
    if (i, j) in [(2, 1), (5, 2)]
        J1h = J1 * bondratio
    end
    if (i, j) in [(6, 1), (3, 2), (3, 1), (6, 2)]
        J1v = J1 * bondratio
    end
    return J1h, J1v
end

# ----------------------------------------------------------------------------
#  Energy for J1J2p on Honeycomb{:brickwall_pi6} (General VUMPS)
# ----------------------------------------------------------------------------
#
# Per-iteration bond assignment (each lattice bond is counted exactly once).
# All J2p bonds live on the EVEN-parity (A) sublattice — this matches the
# "one-triangle" convention of `J1J2p` where J2' couples only A-A sites:
#
#   ─ all (i,j):              J1V   between (i,j)   and (i+1, j)
#   ─ if (i+j) is even (A):   J1H   between (i,j)   and (i,   j+1)
#                             J2\\  between (i,j)   and (i+1, j+1)   (2×2 cluster, A-A)
#                             J2V   between (i,j)   and (i+2, j)     (3-row line, A-A)
#   ─ if (i+j) is odd  (B):   J2/   between (i,j+1) and (i+1, j)     (2×2 cluster, A-A)
#
# The J2V bond replaces the original J2H (skip-1 horizontal) under the
# rows ↔ columns swap. It requires a 3-row vertical contraction (`contract_o3_V`).
# For a unit cell with Ni < 3 the J2V bond wraps onto itself; this is still
# evaluated correctly as a (formal) self-NNN through the periodic image, but
# physically you usually want Ni ≥ 4 (e.g. the Ni=6, Nj=2 analog of the
# standard 2×6 brickwall pattern).

function energy_value(model::J1J2p{Honeycomb{:brickwall_pi6}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2p = model
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    len = length(ACu.data)

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate = false)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy"  => Dict{String, Any}(),
        "bond_J1V_energy"  => Dict{String, Any}(),
        "bond_J2V_energy"  => Dict{String, Any}(),
        "bond_J2\\_energy" => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}(),
    )

    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")

        J1h, J1v = enlarge_coupling(model, i, j)

        # ───── J1V — always (all vertical bonds are non-trivial in pi6) ─────
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21,
            (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j],
             FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]),
            terms; ifparallel, forloop_iter)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j],
                          FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j];
                          ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
        etol += J1v * e/n
        e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

        if (i + j) % 2 == 0
            # ───── J1H — even-parity sites only (right bond is non-trivial) ─────
            # (A-B bond between (i,j) and (i, j+1))
            ird = Ni + 1 - i
            jr  = mod1(j + 1, Nj)
            e = _contract_barebones(contract_o_12,
                (FLo[i,j], ACu[i,j], A[i,j], ACd[ird,j],
                 FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ird,jr]),
                terms; ifparallel, forloop_iter)
            n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ird,j],
                              FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ird,jr];
                              ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
            etol += J1h * e/n
            e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

            # ───── J2\\ — NNN diagonal on the 2×2 cluster anchored at (i,j) ─────
            # A-A bond between (i,j)[even] and (i+1, j+1)[even]
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr  = mod1(j + 1, Nj)
            e1 = _contract_barebones(contract_o_22_1,
                (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j],
                 FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                 A[i,j], A[i,jr], A[ir,j], A[ir,jr]),
                terms_norot; ifparallel, forloop_iter)
            n = contract_n_22(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j],
                              FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                              A[i,j], A[i,jr], A[ir,j], A[ir,jr];
                              ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J2\\ = $(J2p * e1/n)")
            etol += J2p * e1/n
            e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2p * e1/n

            # ───── J2V — skip-1 vertical NNN, 3-row line at column j ─────
            # A-A bond between (i,j)[even] and (i+2, j)[even] via (i+1, j)[odd]
            irr2 = mod1(i + 2, Ni)
            iddr = mod1(Ni - (i + 1), Ni)   # mirror row for ACd at row i+2
            e = _contract_barebones(contract_o3_V,
                (ACu[i,j], ACd[iddr,j],
                 FLu[i,j],  FRu[i,j],
                 FLu[ir,j], FRu[ir,j],
                 FLo[irr2,j], FRo[irr2,j],
                 A[i,j], A[ir,j], A[irr2,j]),
                terms_norot; ifparallel, forloop_iter)
            n = contract_n3_V(ACu[i,j], ACd[iddr,j],
                              FLu[i,j],  FRu[i,j],
                              FLu[ir,j], FRu[ir,j],
                              FLo[irr2,j], FRo[irr2,j],
                              A[i,j], A[ir,j], A[irr2,j];
                              ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J2V = $(J2p * e/n)")
            etol += J2p * e/n
            e_dict["bond_J2V_energy"]["$(i),$(j)"] = J2p * e/n

        else
            # ───── J2/ — NNN diagonal on the 2×2 cluster anchored at odd (i,j) ─────
            # A-A bond between (i, j+1)[even] and (i+1, j)[even]
            # (the two A-corners of a 2×2 cluster whose top-left site is B = odd)
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr  = mod1(j + 1, Nj)
            e2 = _contract_barebones(contract_o_22_2,
                (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j],
                 FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                 A[i,j], A[i,jr], A[ir,j], A[ir,jr]),
                terms_norot; ifparallel, forloop_iter)
            n = contract_n_22(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j],
                              FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                              A[i,j], A[i,jr], A[ir,j], A[ir,jr];
                              ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J2/ = $(J2p * e2/n)")
            etol += J2p * e2/n
            e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2p * e2/n
        end
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end
