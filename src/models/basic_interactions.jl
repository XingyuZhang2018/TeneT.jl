"""
    _contract_barebones(contract_fn, args, terms; kwargs...)

Sum over interaction terms: `Σ c * contract_fn(args..., OL, OR; kwargs...)`.
Each term is `(coefficient, O_left, O_right)` where O_left/O_right are d×d matrices.
"""
function _contract_barebones(contract_fn, args, terms; kwargs...)
    return sum(c * contract_fn(args..., OL, OR; kwargs...) for (c, OL, OR) in terms)
end

# ────────────────────────────────────────────────────────────────────────────
# Checkpointed overloads — pass `params::iPEPSOptimize` to enable
# `params.bond_checkpoint`. Each term's contract_fn call is wrapped in
# `checkpoint(...)` so that during backward only one term's tape is live.
# ifparallel and forloop_iter are read from params, not passed as kwargs.
# ────────────────────────────────────────────────────────────────────────────

"""
    _contract_barebones(contract_fn, args, terms, params::iPEPSOptimize; kwargs...)

Bond-checkpointed `_contract_barebones`. Each `(c, OL, OR)` term is
evaluated under `checkpoint(params.bond_checkpoint, contract_fn, ...)`,
so during backward Zygote re-runs one term's forward at a time, keeping
peak VRAM to a single term's tape instead of the whole sum's tape.
"""
function _contract_barebones(contract_fn, args, terms,
                             params::iPEPSOptimize; kwargs...)
    bond_ckpt    = params.bond_checkpoint
    ifparallel   = params.boundary_alg.ifparallel
    forloop_iter = params.boundary_alg.forloop_iter
    return sum(c * checkpoint(bond_ckpt, contract_fn, args..., OL, OR;
                              ifparallel, forloop_iter, kwargs...)
               for (c, OL, OR) in terms)
end

"""
    _contract_one(contract_fn, args, params::iPEPSOptimize; kwargs...)

Single-contraction checkpointed wrapper, for norms and one-off
observables (`contract_n_12`, `contract_o_11`, ...) that are not summed
over terms but still contribute a large tape entry.
"""
function _contract_one(contract_fn, args::Tuple, params::iPEPSOptimize;
                       kwargs...)
    return checkpoint(params.bond_checkpoint, contract_fn, args...;
                      ifparallel   = params.boundary_alg.ifparallel,
                      forloop_iter = params.boundary_alg.forloop_iter,
                      kwargs...)
end

# Term-by-term interaction decompositions using d×d spin operators (Sp, Sm, Sz).
# Each function returns a list of (coefficient, O_left, O_right) tuples.
"""
    _heisenberg_bond_terms(model, atype; Jx, Jy, Jz, S, ifrotate)

Decompose the Heisenberg interaction into Sp/Sm/Sz terms:
  H = (Jx+Jy)/4*(Sp⊗Sm + Sm⊗Sp) + (Jx-Jy)/4*(Sp⊗Sp + Sm⊗Sm) + Jz*Sz⊗Sz

When `ifrotate=true`, the right operators are rotated by U = 2Sy (sublattice rotation).
"""
function _heisenberg_bond_terms(model::HamiltonianModel, atype;
                                Jx = hasproperty(model, :Jx) ? model.Jx : 1,
                                Jy = hasproperty(model, :Jy) ? model.Jy : 1,
                                Jz = hasproperty(model, :Jz) ? model.Jz : 1,
                                S  = model.S,
                                ifrotate = hasproperty(model, :ifrotate) ? model.ifrotate : false)
    Sp = atype(const_Sp(S))
    Sm = atype(const_Sm(S))
    Sz = atype(const_Sz(S))
    if ifrotate
        U = atype(2 * const_Sy(S))
        rSp = real(U * Sp * U')
        rSm = real(U * Sm * U')
        rSz = real(U * Sz * U')
    else
        rSp, rSm, rSz = Sp, Sm, Sz
    end
    Jpm = (Jx + Jy) / 4
    Jpp = (Jx - Jy) / 4
    terms = [(Jpm, Sp, rSm), (Jpm, Sm, rSp), (Jz, Sz, rSz)]
    if Jpp != 0
        push!(terms, (Jpp, Sp, rSp))
        push!(terms, (Jpp, Sm, rSm))
    end
    return terms
end

"""
    _kitaev_bond_terms(bond_type, S, atype; ifrotate=false)

Return `(O_left, O_right)` operator pair for a single Kitaev bond.
Uses Sx/Sy/Sz directly (Sy bond involves complex arithmetic but only 1 contraction).
"""
function _kitaev_bond_terms(bond_type::Symbol, S, atype; ifrotate=false)
    Sx = atype(const_Sx(S))
    Sy = atype(ComplexF64.(const_Sy(S)))
    Sz = atype(const_Sz(S))
    if ifrotate
        U = atype(2 * const_Sy(S))
        rSx = real(U * Sx * U')
        rSy = U * Sy * U'
        rSz = real(U * Sz * U')
    else
        rSx, rSy, rSz = Sx, Sy, Sz
    end
    if bond_type == :x
        return Sx, rSx
    elseif bond_type == :y
        return Sy, rSy
    elseif bond_type == :z
        return Sz, rSz
    else
        error("Unknown Kitaev bond type: $bond_type")
    end
end

# ── Kagome merge: d³×d³ operator construction from Heisenberg terms ──

"""
    _kagome_onsite_op(terms, pos1, pos2, d, atype)

Build d³×d³ onsite operator for intra-cell bond between sublattice `pos1` and `pos2`.
`pos1` is where OL acts, `pos2` is where OR acts (both ∈ {1,2,3}).
`terms` is a list of `(coeff, OL, OR)` from `_heisenberg_bond_terms`.
"""
function _kagome_onsite_op(terms, pos1, pos2, d, atype)
    Id = Matrix{Float64}(I, d, d)
    h = zeros(Float64, d^3, d^3)
    for (c, OL, OR) in terms
        ops = [Id, Id, Id]
        ops[pos1] = Array(OL)
        ops[pos2] = Array(OR)
        @tensor o[a,b,c,d,e,f] := ops[1][a,d] * ops[2][b,e] * ops[3][c,f]
        h += c * reshape(real(o), d^3, d^3)
    end
    return atype(h)
end

"""
    _kagome_intercell_terms(terms, sublattice_left, sublattice_right, d, atype)

Build list of `(coeff, OL_d³, OR_d³)` for inter-cell bond.
`sublattice_left/right ∈ {1,2,3}` — which sublattice the operator acts on in each cell.
"""
function _kagome_intercell_terms(terms, sublattice_left, sublattice_right, d, atype)
    Id = Matrix{Float64}(I, d, d)
    result = Tuple{Real, AbstractMatrix, AbstractMatrix}[]
    for (c, OL, OR) in terms
        ops_l = [Id, Id, Id]; ops_l[sublattice_left] = Array(OL)
        ops_r = [Id, Id, Id]; ops_r[sublattice_right] = Array(OR)
        @tensor ol[a,b,c,d,e,f] := ops_l[1][a,d] * ops_l[2][b,e] * ops_l[3][c,f]
        @tensor or_t[a,b,c,d,e,f] := ops_r[1][a,d] * ops_r[2][b,e] * ops_r[3][c,f]
        push!(result, (c, atype(real(reshape(ol, d^3, d^3))), atype(real(reshape(or_t, d^3, d^3)))))
    end
    return result
end

"""
    _kagome_site_op(O, sublattice, d)

Embed a d×d operator into d³×d³ space at the given sublattice position.
"""
function _kagome_site_op(O, sublattice, d)
    Id = Matrix{Float64}(I, d, d)
    ops = [Id, Id, Id]
    ops[sublattice] = Array(O)
    @tensor out[a,b,c,d,e,f] := ops[1][a,d] * ops[2][b,e] * ops[3][c,f]
    return reshape(real(out), d^3, d^3)
end
