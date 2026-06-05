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
        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), params)
        params.verbosity >= 4 && println("bond_H = $(e/n)")
        etol += e/n
        e_dict["bond_H_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms, params)
        n = _contract_one(contract_n_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), params)
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
        e = _contract_barebones(contract_o_12, (FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr]), params)
        params.verbosity >= 4 && println("bond_H = $(e/n)")
        etol += e/n
        e_dict["bond_H_energy"]["$(i),$(j)"] = e/n

        e = _contract_barebones(contract_o_21, (AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j]), terms, params)
        n = _contract_one(contract_n_21, (AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j]), params)
        params.verbosity >= 4 && println("bond_V = $(e/n)")
        etol += e/n
        e_dict["bond_V_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::C4vVUMPSEnv, params::iPEPSOptimize)
    @unpack AL, C, FL = env
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_H_energy" => Dict{String, Any}(),
    )

    A1 = A[1]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)
    AC = ALCtoAC_map(AL,C)

    e = _contract_barebones(contract_o_12, (FL, AL, A1, conj(AL), FL, AC, A1, conj(AC)), terms, params)
    n = _contract_one(contract_n_12, (FL, AL, A1, conj(AL), FL, AC, A1, conj(AC)), params)
    params.verbosity >= 4 && println("bond_H = $(e/n)")
    etol = e/n
    e_dict["bond_H_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::CTMEnv, params::iPEPSOptimize)
    @unpack C, T = env
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_H_energy" => Dict{String, Any}(),
    )

    A1 = A[1]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)

    To = CTCtoT(C, T)
    e = _contract_barebones(contract_o_12, (To, T, A1, T, To, T, A1, T), terms, params)
    n = _contract_one(contract_n_12, (To, T, A1, T, To, T, A1, T), params)
    params.verbosity >= 4 && println("bond_H = $(e/n)")
    etol = e/n
    e_dict["bond_H_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::Heisenberg{Honeycomb{:c3v}}, A, env::CTMEnv, params::iPEPSOptimize)
    @unpack C, T = env
    A1 = A[1]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)

    e = _contract_barebones(_contract_c3v_bond, (C, T, A1), terms, params)
    n = _contract_one(_contract_c3v_bond_norm, (C, T, A1), params)
    e_bond = e / n
    etol = 3 * e_bond / 2

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_C3v_energy" => Dict("1,1" => e_bond),
    )

    params.verbosity >= 3 && println("energy per site = $(etol)")
    return etol, e_dict
end

function energy_value(model::Heisenberg{Honeycomb{:c3v}}, A, env::Tuple{CTMEnv, CTMEnv}, params::iPEPSOptimize)
    env1, env2 = env
    C1, T1 = env1.C, env1.T
    C2, T2 = env2.C, env2.T
    A1 = A[1]
    A2 = A[2]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)

    e = _contract_barebones(_contract_c3v_bond, (C1, T1, C2, T2, A1, A2), terms, params)
    n = _contract_one(_contract_c3v_bond_norm, (C1, T1, C2, T2, A1, A2), params)
    e_bond = e / n
    etol = 3 * e_bond / 2

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_C3v_energy" => Dict("1,1" => e_bond),
    )

    params.verbosity >= 3 && println("energy per site = $(etol)")
    return etol, e_dict
end


function energy_value(model::Heisenberg{Honeycomb{:brickwall_h}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
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
            e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms, params)
            n = _contract_one(contract_n_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), params)
            params.verbosity >= 4 && println("bond_J1V = $(e/n)")
            etol += e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] =  e/n
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), params)
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
    model.ifrotate && throw(ArgumentError("Kagome merge does not support ifrotate=true"))
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
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
        n = _contract_one(contract_n_11, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]), params)

        e = _contract_one(contract_o_11, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], h_12), params)
        e_dict["bond_12_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_12 = $(e/n)")

        e = _contract_one(contract_o_11, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], h_23), params)
        e_dict["bond_23_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_23 = $(e/n)")

        # Inter-cell horizontal bonds
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        n = _contract_one(contract_n_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), params)

        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms_31H, params)
        e_dict["bond_31H_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_31H = $(e/n)")

        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms_32H, params)
        e_dict["bond_32H_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_32H = $(e/n)")

        # Inter-cell vertical bonds
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        n = _contract_one(contract_n_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), params)

        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms_31V, params)
        e_dict["bond_31V_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_31V = $(e/n)")

        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms_21V, params)
        e_dict["bond_21V_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_21V = $(e/n)")
    end

    return e_dict
end

function energy_value(model::Heisenberg{Kagome{:merge}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    model.ifrotate && throw(ArgumentError("Kagome merge does not support ifrotate=true"))
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
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
        e = _contract_one(contract_o_11, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j], h_onsite), params)
        n = _contract_one(contract_n_11, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j]), params)
        params.verbosity >= 4 && println("bond_onsite = $(e/n)")
        etol += e/n
        e_dict["bond_onsite_energy"]["$(i),$(j)"] = e/n

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms_H, params)
        n = _contract_one(contract_n_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), params)
        params.verbosity >= 4 && println("bond_H = $(e/n)")
        etol += e/n
        e_dict["bond_H_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms_V, params)
        n = _contract_one(contract_n_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), params)
        params.verbosity >= 4 && println("bond_V = $(e/n)")
        etol += e/n
        e_dict["bond_V_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len/3)")
    return etol/len/3, e_dict
end

"""
    energy_value(model::Heisenberg{<:KagomeOnehole}, A, env::VUMPSEnv, params)

Setup (a) Kagome embedding (works for both `:onehole` and `:onehole_real`):
three physical sites + one empty/δ per 2×2 sub-block. Pattern must be
(2N)×(2M); role of each tensor is determined by (i, j) parity:
  (odd, odd) = A    (even, odd) = B    (odd, even) = C    (even, even) = empty
Six Kagome bonds per 2×2 sub-block, distributed across the four sites by
ownership. The empty site carries no Hamiltonian operator (its physical
leg is summed in all bond contractions).
"""
function energy_value(model::Heisenberg{<:KagomeOnehole}, A, env::VUMPSEnv, params::iPEPSOptimize)
    model.ifrotate && throw(ArgumentError("Kagome :onehole does not support ifrotate=true"))
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(A)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("Pattern must be (2N)x(2M) for Kagome :onehole; got ($Ni,$Nj)"))
    atype = _arraytype(A[1])
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_AC_H_energy"  => Dict{String, Any}(),  # bond 1
        "bond_AB_V_energy"  => Dict{String, Any}(),  # bond 2
        "bond_BC_diag_energy" => Dict{String, Any}(),  # bond 3
        "bond_CA_H_cross_energy"  => Dict{String, Any}(),  # bond 4
        "bond_BA_V_cross_energy"  => Dict{String, Any}(),  # bond 5
        "bond_BC_diag_cross_energy" => Dict{String, Any}(),  # bond 6
    )

    # Bare d×d Heisenberg terms (no rotation, no d^3 promotion)
    terms = _heisenberg_bond_terms(model, atype; ifrotate=false)

    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        isA     = (i % 2 == 1) && (j % 2 == 1)
        isB     = (i % 2 == 0) && (j % 2 == 1)
        isC     = (i % 2 == 1) && (j % 2 == 0)
        isE     = (i % 2 == 0) && (j % 2 == 0)

        if isA
            # Index conventions for this branch:
            #   ir         = Ni + 1 - i       — reflected row for ACd (used by contract_o_12 / contract_o_22_2)
            #   jr         = mod1(j + 1, Nj)  — column-shifted neighbor (bonds 1, 3)
            #   ir2 / irr2 = mod1(i + 1, Ni) / mod1(Ni - i, Ni) — row-shifted neighbor (bonds 2, 3)
            ir   = Ni + 1 - i
            jr   = mod1(j + 1, Nj)
            ir2  = mod1(i + 1, Ni)
            irr2 = mod1(Ni - i, Ni)

            # Bond 1: A–C horizontal NN
            e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms, params)
            n = _contract_one(contract_n_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), params)
            params.verbosity >= 4 && println("bond_AC_H = $(e/n)")
            etol += e/n
            e_dict["bond_AC_H_energy"]["$(i),$(j)"] = e/n

            # Bond 2: A–B vertical NN
            e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir2,j], A[ir2,j], FRo[ir2,j], ACd[irr2,j]), terms, params)
            n = _contract_one(contract_n_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir2,j], A[ir2,j], FRo[ir2,j], ACd[irr2,j]), params)
            params.verbosity >= 4 && println("bond_AB_V = $(e/n)")
            etol += e/n
            e_dict["bond_AB_V_energy"]["$(i),$(j)"] = e/n

            # Bond 3: B–C anti-diagonal plaquette at (i,j); operators on (i,jr) (=C) and (ir2,j) (=B)
            e = _contract_barebones(contract_o_22_2,
                (FLu[i,j], FLo[ir2,j], ACu[i,j], ACd[irr2,j], FRu[i,jr], FRo[ir2,jr], ARu[i,jr], ARd[irr2,jr],
                 A[i,j], A[i,jr], A[ir2,j], A[ir2,jr]),
                terms, params)
            n = _contract_one(contract_n_22,
                (FLu[i,j], FLo[ir2,j], ACu[i,j], ACd[irr2,j], FRu[i,jr], FRo[ir2,jr], ARu[i,jr], ARd[irr2,jr],
                 A[i,j], A[i,jr], A[ir2,j], A[ir2,jr]),
                params)
            params.verbosity >= 4 && println("bond_BC_diag = $(e/n)")
            etol += e/n
            e_dict["bond_BC_diag_energy"]["$(i),$(j)"] = e/n
        end

        if isC
            # Bond 4: C–A horizontal NN to next cell
            ir = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms, params)
            n = _contract_one(contract_n_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), params)
            params.verbosity >= 4 && println("bond_CA_H = $(e/n)")
            etol += e/n
            e_dict["bond_CA_H_cross_energy"]["$(i),$(j)"] = e/n
        end

        if isB
            # Bond 5: B–A vertical NN to next cell down
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms, params)
            n = _contract_one(contract_n_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), params)
            params.verbosity >= 4 && println("bond_BA_V = $(e/n)")
            etol += e/n
            e_dict["bond_BA_V_cross_energy"]["$(i),$(j)"] = e/n
        end

        if isE
            # Bond 6: B'–C' anti-diagonal plaquette at (i,j); plaquette spans
            # {empty, B', C', A''} across 4 neighboring sub-blocks.
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e = _contract_barebones(contract_o_22_2,
                (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                 A[i,j], A[i,jr], A[ir,j], A[ir,jr]),
                terms, params)
            n = _contract_one(contract_n_22,
                (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                 A[i,j], A[i,jr], A[ir,j], A[ir,jr]),
                params)
            params.verbosity >= 4 && println("bond_BC_diag_cross = $(e/n)")
            etol += e/n
            e_dict["bond_BC_diag_cross_energy"]["$(i),$(j)"] = e/n
        end
    end

    params.verbosity >= 4 && println("energy = $(etol/len*4/3)")
    return etol/len*4/3, e_dict
end
