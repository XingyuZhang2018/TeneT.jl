export J1J2J3

"""
    J1J2J3{L<:AbstractLattice}

J1-J2-J3 Heisenberg model with nearest-neighbor coupling `J1` and next-nearest-neighbor coupling `J2` and next-next-nearest-neighbor coupling `J3`. The model is defined on a lattice `L`, which can be specified by the user (e.g., `Square()`, `Honeycomb(:brickwall_h)`, etc.). The spin magnitude is given by `S`. The parameter `ifrotate` determines whether to rotate the spin operators in the Hamiltonian, and `couplingtype` specifies the type of coupling pattern (e.g., uniform, plaquette, dimmer, etc.). The `bondratio` parameter is used to adjust the coupling strength for non-uniform patterns.
"""
@kwdef mutable struct J1J2J3{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.6
    J3::Real = 0.4
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette, :dimmer1, :dimmer2, :mixed
    bondratio::Real = 1.0 # only used when couplingtype is not :uniform
                          # for Honeycomb{:brickwall_h}, bondratio<1 is plaquette, bondratio>1 is dimmer
end

function energy_value(model::J1J2J3{Honeycomb{:brickwall_h}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2, J3 = model
    atype = _arraytype(ACu[1])
    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    Sp = Zygote.@ignore atype(const_Sp(model.S))
    Sm = Zygote.@ignore atype(const_Sm(model.S))
    Sz = Zygote.@ignore atype(const_Sz(model.S))
    d = size(Sz, 1)
    Id = Zygote.@ignore atype(atype(Matrix{Float64}(I, d, d)))

    # Rotated operators for the even sublattice (U = 2Sy: Sp→-Sm, Sm→-Sp, Sz→-Sz)
    if model.ifrotate
        rSp, rSm, rSz = -Sm, -Sp, -Sz
    else
        rSp, rSm, rSz = Sp, Sm, Sz
    end

    Ni, Nj = size(ACu)
    len = length(ACu.data)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J2H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
        "bond_J2\\_energy"  => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}(),
        "bond_J3\\_energy"  => Dict{String, Any}(),
        "bond_J3/_energy"  => Dict{String, Any}(),
        "bond_J3|_energy"  => Dict{String, Any}()
    )
    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms, params)
        n = _contract_one(contract_n_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), params)
        if (i + j) % 2 != 0
            params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
            etol += J1v * e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

            ir = mod1(i + 1, Ni)
            id = mod1(Ni - i, Ni) 
            jr = mod1(j + 1, Nj)
            jrr = mod1(j + 2, Nj)

            n = _contract_one(contract_n_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr]), params)

            # J3\\ bond: site 1=(i,j) unrotated, site 6=(ir,jrr) on rotated sublattice
            e = 1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Sp, Id, Id, Id, Id, rSm), params) +
            1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Sm, Id, Id, Id, Id, rSp), params) +
            _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Sz, Id, Id, Id, Id, rSz), params)
            params.verbosity >= 4 && println("bond_J3\\ = $(J3 * e/n)")
            etol += J3 * e/n
            e_dict["bond_J3\\_energy"]["$(i),$(j)"] = J3 * e/n

            # J3/ bond: site 3=(i,jr) on rotated sublattice, site 4=(ir,jr) unrotated
            e = 1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Id, Id, rSp, Sm, Id, Id), params) +
            1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Id, Id, rSm, Sp, Id, Id), params) +
            _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Id, Id, rSz, Sz, Id, Id), params)
            params.verbosity >= 4 && println("bond_J3/ = $(J3 * e/n)")
            etol += J3 * e/n
            e_dict["bond_J3/_energy"]["$(i),$(j)"] = J3 * e/n
        else
            params.verbosity >= 4 && println("bond_J3| = $(J1v * e/n)")
            etol += J3 * e/n
            e_dict["bond_J3|_energy"]["$(i),$(j)"] = J3 * e/n
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
        e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
        n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e1/n)")
        params.verbosity >= 4 && println("bond_J2/ = $(J2 * e2/n)")
        etol += J2 * (e1 + e2)/n
        e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2 * e2/n

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        jrr = mod1(j + 2, Nj)
        e = _contract_barebones(contract_o_13, (FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), terms_norot, params)
        n = _contract_one(contract_n_13, (FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), params)
        params.verbosity >= 4 && println("bond_J2H = $(J2 * e/n)")
        etol += J2 * e/n
        e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2 * e/n
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::J1J2J3{Honeycomb{:brickwall_h}}, A, env::PlaquetteVUMPSEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for Plaquette VUMPS energy evaluation"))

    @unpack AL, C, FLu, FLo = env
    @unpack J2, J3 = model
    AC = ALCtoAC(AL, C)

    atype = _arraytype(A[1])
    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    Sp = Zygote.@ignore atype(const_Sp(model.S))
    Sm = Zygote.@ignore atype(const_Sm(model.S))
    Sz = Zygote.@ignore atype(const_Sz(model.S))
    d = size(Sz, 1)
    Id = Zygote.@ignore atype(atype(Matrix{Float64}(I, d, d)))

    # Rotated operators for the even sublattice (U = 2Sy: Sp→-Sm, Sm→-Sp, Sz→-Sz)
    rSp, rSm, rSz = -Sm, -Sp, -Sz   # ifrotate is always true for Plaquette

    Ni, Nj = size(A)
    len = length(A.data)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J2H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
        "bond_J2\\_energy"  => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}(),
        "bond_J3\\_energy"  => Dict{String, Any}(),
        "bond_J3/_energy"  => Dict{String, Any}(),
        "bond_J3|_energy"  => Dict{String, Any}()
    )
    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jo = mod1(Nj - j, Nj)
        e = _contract_barebones(contract_o_21, (AC[i,j],FLu[i,j],A[i,j],FLu[i,jo],FLo[ir,j],A[ir,j],FLo[ir,jo],AC[irr,j]), terms, params)
        n = _contract_one(contract_n_21, (AC[i,j],FLu[i,j],A[i,j],FLu[i,jo],FLo[ir,j],A[ir,j],FLo[ir,jo],AC[irr,j]), params)
        if (i + j) % 2 != 0
            params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
            etol += J1v * e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

            ir = mod1(i + 1, Ni)
            id = mod1(Ni - i, Ni) 
            jr = mod1(j + 1, Nj)
            jrr = mod1(j + 2, Nj)
            jo = mod1(Nj - (j+2), Nj)
            n = _contract_one(contract_n_23, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AL[i,jr], AL[id,jr], AC[i,jrr], AC[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr]), params)

            # J3\\ bond: site 1=(i,j) unrotated, site 6=(ir,jrr) on rotated sublattice
            e = 1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AL[i,jr], AL[id,jr], AC[i,jrr], AC[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Sp, Id, Id, Id, Id, rSm), params) +
            1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AL[i,jr], AL[id,jr], AC[i,jrr], AC[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Sm, Id, Id, Id, Id, rSp), params) +
            _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AL[i,jr], AL[id,jr], AC[i,jrr], AC[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Sz, Id, Id, Id, Id, rSz), params)
            params.verbosity >= 4 && println("bond_J3\\ = $(J3 * e/n)")
            etol += J3 * e/n
            e_dict["bond_J3\\_energy"]["$(i),$(j)"] = J3 * e/n

            # J3/ bond: site 3=(i,jr) on rotated sublattice, site 4=(ir,jr) unrotated
            e = 1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AL[i,jr], AL[id,jr], AC[i,jrr], AC[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Id, Id, rSp, Sm, Id, Id), params) +
            1/2 * _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AL[i,jr], AL[id,jr], AC[i,jrr], AC[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Id, Id, rSm, Sp, Id, Id), params) +
            _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AL[i,jr], AL[id,jr], AC[i,jrr], AC[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], Id, Id, rSz, Sz, Id, Id), params)
            params.verbosity >= 4 && println("bond_J3/ = $(J3 * e/n)")
            etol += J3 * e/n
            e_dict["bond_J3/_energy"]["$(i),$(j)"] = J3 * e/n
        else
            params.verbosity >= 4 && println("bond_J3| = $(J1v * e/n)")
            etol += J3 * e/n
            e_dict["bond_J3|_energy"]["$(i),$(j)"] = J3 * e/n
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        jo = mod1(Nj - (j+1), Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j],AL[i,j],A[i,j],AL[ir,j],FLo[i,jo],AC[i,jr],A[i,jr],AC[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j],AL[i,j],A[i,j],AL[ir,j],FLo[i,jo],AC[i,jr],A[i,jr],AC[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        ir  = mod1(i + 1, Ni)
        id = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        jo = mod1(Nj - (j+1), Nj)
        e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AC[i,jr], AC[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
        e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AC[i,jr], AC[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
        n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AC[i,jr], AC[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e1/n)")
        params.verbosity >= 4 && println("bond_J2/ = $(J2 * e2/n)")
        etol += J2 * (e1 + e2)/n
        e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2 * e2/n

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        jrr = mod1(j + 2, Nj)
        jo = mod1(Nj - (j+2), Nj)
        e = _contract_barebones(contract_o_13, (FLo[i,j], AL[i,j], AL[ir,j], FLo[i,jo], AL[i,jr], AL[ir,jr], AC[i,jrr], AC[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), terms_norot, params)
        n = _contract_one(contract_n_13, (FLo[i,j], AL[i,j], AL[ir,j], FLo[i,jo], AL[i,jr], AL[ir,jr], AC[i,jrr], AC[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), params)
        params.verbosity >= 4 && println("bond_J2H = $(J2 * e/n)")
        etol += J2 * e/n
        e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2 * e/n
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end