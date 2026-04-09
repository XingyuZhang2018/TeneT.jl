export Heisenberg

"""
    Heisenberg{L}(; lattice, S, Jx, Jy, Jz)

Heisenberg model with couplings `Jx`, `Jy`, `Jz` on a given lattice.
"""
@kwdef mutable struct Heisenberg{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette...
    bondratio = 1.0
end

function energy_value(model::Heisenberg{Square}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    terms = _heisenberg_bond_terms(model, atype)
    etol = 0
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_H_energy" => Dict{String, Any}(),
        "bond_V_energy"   => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms; forloop_iter, ifparallel)
        n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter, ifparallel)
        params.verbosity >= 4 && println("bond_H = $(e/n)")
        etol += e/n
        e_dict["bond_H_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms; forloop_iter, ifparallel)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter, ifparallel)
        params.verbosity >= 4 && println("bond_V = $(e/n)")
        etol += e/n
        e_dict["bond_V_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::PlaquetteVUMPSEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for Plaquette VUMPS energy evaluation"))
    
    @unpack AL, C, FLu, FLo = env
    AC = ALCtoAC(AL, C)
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    terms = _heisenberg_bond_terms(model, atype)
    etol = 0
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_H_energy" => Dict{String, Any}(),
        "bond_V_energy"   => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        ir = mod1(i + 1, Ni)
        jr = mod1(j + 1, Nj)

        params.verbosity >= 4 && println("===========$i,$j===========")
        e = _contract_barebones(contract_o_12, (FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr]), terms; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_H = $(e/n)")
        etol += e/n
        e_dict["bond_H_energy"]["$(i),$(j)"] = e/n

        e = _contract_barebones(contract_o_21, (AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j]), terms; ifparallel, forloop_iter)
        n = contract_n_21(AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_V = $(e/n)")
        etol += e/n
        e_dict["bond_V_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::C4vVUMPSEnv, params::iPEPSOptimize)
    @unpack AL, C, FL = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_H_energy" => Dict{String, Any}(),
    )

    A1 = A[1]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)
    AC = ALCtoAC_map(AL,C)

    e = _contract_barebones(contract_o_12, (FL, AL, A1, conj(AL), FL, AC, A1, conj(AC)), terms; ifparallel, forloop_iter)
    n = contract_n_12(FL, AL, A1, conj(AL), FL, AC, A1, conj(AC); ifparallel, forloop_iter)
    params.verbosity >= 4 && println("bond_H = $(e/n)")
    etol = e/n
    e_dict["bond_H_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::CTMEnv, params::iPEPSOptimize)
    @unpack C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_H_energy" => Dict{String, Any}(),
    )

    A1 = A[1]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)

    To = CTCtoT(C, T)
    e = _contract_barebones(contract_o_12, (To, T, A1, T, To, T, A1, T), terms; ifparallel, forloop_iter)
    n = contract_n_12(To, T, A1, T, To, T, A1, T; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("bond_H = $(e/n)")
    etol = e/n
    e_dict["bond_H_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::Heisenberg{Honeycomb{:brickwall}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    len = length(ACu.data)

    terms = _heisenberg_bond_terms(model, atype)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
    )
    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms; ifparallel, forloop_iter)
            n = contract_n_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J1V = $(e/n)")
            etol += e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] =  e/n
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_J1H = $(e/n)")
        etol += e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end

"""
    energy_value_perbond(model::Heisenberg{Kagome{:merge}}, A, env, params)

Compute per-bond energies for Kagome merge (6 bonds per unit cell).
"""
function energy_value_perbond(model::Heisenberg{Kagome{:merge}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    @unpack Jx, Jy, Jz, S = model

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    len = length(A)

    d = Int(2*S + 1)
    # Kagome merge: no sublattice rotation
    terms = _heisenberg_bond_terms(model, Array; ifrotate=false)

    h_12 = _kagome_onsite_op(terms, 1, 2, d, atype)
    h_23 = _kagome_onsite_op(terms, 2, 3, d, atype)

    # Inter-cell operator term lists (d³×d³ pairs)
    terms_31H = _kagome_intercell_terms(terms, 3, 1, d, atype)  # site 3@left ↔ site 1@right
    terms_32H = _kagome_intercell_terms(terms, 3, 2, d, atype)  # site 3@left ↔ site 2@right
    terms_31V = _kagome_intercell_terms(terms, 3, 1, d, atype)  # site 3@upper ↔ site 1@lower
    terms_21V = _kagome_intercell_terms(terms, 2, 1, d, atype)  # site 2@upper ↔ site 1@lower

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_12_energy"  => Dict{String, Any}(),
        "bond_23_energy"  => Dict{String, Any}(),
        "bond_31H_energy" => Dict{String, Any}(),
        "bond_32H_energy" => Dict{String, Any}(),
        "bond_31V_energy" => Dict{String, Any}(),
        "bond_21V_energy" => Dict{String, Any}()
    )

    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")

        # Intra-cell bonds: onsite contraction
        ir = Ni + 1 - i
        n = contract_n_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]; ifparallel, forloop_iter)

        e = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], h_12; ifparallel, forloop_iter)
        e_dict["bond_12_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_12 = $(e/n)")

        e = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], h_23; ifparallel, forloop_iter)
        e_dict["bond_23_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_23 = $(e/n)")

        # Inter-cell horizontal bonds
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; ifparallel, forloop_iter)

        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms_31H; ifparallel, forloop_iter)
        e_dict["bond_31H_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_31H = $(e/n)")

        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms_32H; ifparallel, forloop_iter)
        e_dict["bond_32H_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_32H = $(e/n)")

        # Inter-cell vertical bonds
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; ifparallel, forloop_iter)

        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms_31V; ifparallel, forloop_iter)
        e_dict["bond_31V_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_31V = $(e/n)")

        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms_21V; ifparallel, forloop_iter)
        e_dict["bond_21V_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_21V = $(e/n)")
    end

    return e_dict
end

function energy_value(model::Heisenberg{Kagome{:merge}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_onsite_energy"   => Dict{String, Any}(),
        "bond_H_energy" => Dict{String, Any}(),
        "bond_V_energy"   => Dict{String, Any}()
    )

    d = Int(2*model.S + 1)
    # Kagome merge: no sublattice rotation (all sites in same cell)
    terms = _heisenberg_bond_terms(model, Array; ifrotate=false)

    h_onsite = _kagome_onsite_op(terms, 1, 2, d, atype) + _kagome_onsite_op(terms, 2, 3, d, atype)

    # Inter-cell terms: H combines 31H + 32H, V combines 31V + 21V
    terms_H = vcat(
        _kagome_intercell_terms(terms, 3, 1, d, atype),
        _kagome_intercell_terms(terms, 3, 2, d, atype)
    )
    terms_V = vcat(
        _kagome_intercell_terms(terms, 3, 1, d, atype),
        _kagome_intercell_terms(terms, 2, 1, d, atype)
    )

    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")

        ir = Ni + 1 - i
        e = contract_o_11(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j], h_onsite; ifparallel, forloop_iter)
        n = contract_n_11(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_onsite = $(e/n)")
        etol += e/n
        e_dict["bond_onsite_energy"]["$(i),$(j)"] = e/n

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms_H; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_H = $(e/n)")
        etol += e/n
        e_dict["bond_H_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms_V; ifparallel, forloop_iter)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_V = $(e/n)")
        etol += e/n
        e_dict["bond_V_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end
# ── Kagome merge: hamiltonian for SU_parameterization ──

"""
    hamiltonian(model::Heisenberg{Kagome{:merge}})

Return `(h_H, h_V)` inter-cell Hamiltonians as d³×d³×d³×d³ tensors.
h_H = bond(3→1) + bond(3→2), h_V = bond(3→1) + bond(2→1).
"""
function hamiltonian(model::Heisenberg{Kagome{:merge}})
    S = model.S
    d = Int(2*S + 1)
    terms = _heisenberg_bond_terms(model, Array; ifrotate=false)

    function _build_twosite(sublattice_left, sublattice_right)
        h = zeros(Float64, d^3, d^3, d^3, d^3)
        for (c, OL, OR) in terms
            OL_d3 = _kagome_site_op(OL, sublattice_left, d)
            OR_d3 = _kagome_site_op(OR, sublattice_right, d)
            @tensor o[a,b,c,d] := OL_d3[a,b] * OR_d3[c,d]
            h += c * real(o)
        end
        return h
    end

    h_H = _build_twosite(3, 1) + _build_twosite(3, 2)
    h_V = _build_twosite(3, 1) + _build_twosite(2, 1)
    return h_H, h_V
end
