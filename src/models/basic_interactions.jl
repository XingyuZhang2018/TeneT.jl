"""
    _contract_barebones(contract_fn, args, terms; kwargs...)

Sum over interaction terms: `Σ c * contract_fn(args..., OL, OR; kwargs...)`.
Each term is `(coefficient, O_left, O_right)` where O_left/O_right are d×d matrices.
"""
function _contract_barebones(contract_fn, args, terms; kwargs...)
    return sum(c * contract_fn(args..., OL, OR; kwargs...) for (c, OL, OR) in terms)
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

# ── Generic hamiltonian interface (used by SU_parameterization) ──

"""
    hamiltonian(model::HamiltonianModel)

Return the two-site Hamiltonian as a d×d×d×d tensor, constructed from bond terms.
"""
function hamiltonian(model::HamiltonianModel)
    S = model.S
    d = Int(2*S + 1)
    terms = _heisenberg_bond_terms(model, Array)
    h = zeros(Float64, d, d, d, d)
    for (c, OL, OR) in terms
        @tensor o[i,j,k,l] := OL[i,j] * OR[k,l]
        h += c * real(o)
    end
    return h
end

"""
    hamiltonian_onsite(model)

Return the d³×d³ onsite Hamiltonian for Kagome merge (bond 12 + bond 23).
"""
function hamiltonian_onsite(model::HamiltonianModel)
    S = model.S
    d = Int(2*S + 1)
    terms = _heisenberg_bond_terms(model, Array)
    return _kagome_onsite_op(terms, 1, 2, d, Array) + _kagome_onsite_op(terms, 2, 3, d, Array)
end
