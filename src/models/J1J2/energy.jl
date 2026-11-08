export J1J2

"""
    J1J2{L<:AbstractLattice}

J1-J2 Heisenberg model with nearest-neighbor coupling `J1` and next-nearest-neighbor coupling `J2` on a given lattice.
"""
@kwdef mutable struct J1J2{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.5
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette, :dimmer1, :dimmer2, :mixed
    bondratio::Real = 1.0 # only used when couplingtype is not :uniform
                          # for Honeycomb{:brickwall_h}, bondratio<1 is plaquette, bondratio>1 is dimmer
end

_supports_dist_energy_general(::J1J2{Square}) = true
_supports_dist_energy_general(::J1J2{Honeycomb{:brickwall_h}}) = true

"""
    energy_value(model::J1J2{Honeycomb{:merge}}, A, env::VUMPSEnv, params)

Two-site honeycomb merge geometry on an effective square lattice. Each square
cell contains two honeycomb sublattice sites on one merged physical leg:
`1(cell)-2(cell)` is the intra-cell J1 bond, while `2(cell)-1(right)` and
`2(cell)-1(down)` are the inter-cell J1 bonds. Unlike J1J2p, J2 is placed on
both triangular sublattices.
"""
function energy_value(model::J1J2{Honeycomb{:merge}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    model.ifrotate && throw(ArgumentError("J1J2{Honeycomb{:merge}} supports only ifrotate=false."))
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2 = model

    Ni, Nj = size(A)
    len = length(A)
    atype = _arraytype(A[1])
    d = Int(2 * model.S + 1)
    size(A[1], 5) == d^2 ||
        throw(ArgumentError("Honeycomb{:merge} expects merged physical dimension d^2=$(d^2); got $(size(A[1], 5))."))

    terms = _heisenberg_bond_terms(model, Array; ifrotate=false)
    h_J1_onsite = _honeycomb_merge_onsite_op(terms, 1, 2, d, atype)
    terms_J1_inter = _honeycomb_merge_intercell_terms(terms, 2, 1, d, atype)
    terms_J2_1 = _honeycomb_merge_intercell_terms(terms, 1, 1, d, atype)
    terms_J2_2 = _honeycomb_merge_intercell_terms(terms, 2, 2, d, atype)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1_onsite_energy" => Dict{String, Any}(),
        "bond_J1H_energy"       => Dict{String, Any}(),
        "bond_J1V_energy"       => Dict{String, Any}(),
        "bond_J2H_energy"       => Dict{String, Any}(),
        "bond_J2V_energy"       => Dict{String, Any}(),
        "bond_J2/_energy"       => Dict{String, Any}(),
        "bond_J2H2_energy"      => Dict{String, Any}(),
        "bond_J2V2_energy"      => Dict{String, Any}(),
        "bond_J2/2_energy"      => Dict{String, Any}()
    )

    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        J1o, J1h, J1v = enlarge_coupling(model, i, j)

        # J1 onsite: sublattice 1 and 2 in the same merged cell.
        ir = Ni + 1 - i
        args11 = (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j])
        e = _contract_one(contract_o_11, (args11..., h_J1_onsite), params)
        n = _contract_one(contract_n_11, args11, params)
        params.verbosity >= 4 && println("bond_J1_onsite = $(J1o * e/n)")
        etol += J1o * e/n
        e_dict["bond_J1_onsite_energy"]["$(i),$(j)"] = J1o * e/n

        # Horizontal inter-cell bonds:
        # J1H: 2(cell) - 1(right); J2H: same-sublattice bonds on 1 and 2.
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        args12 = (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j],
                  FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr])
        n12 = _contract_one(contract_n_12, args12, params)

        e = _contract_barebones(contract_o_12, args12, terms_J1_inter, params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n12)")
        etol += J1h * e/n12
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n12

        e1 = _contract_barebones(contract_o_12, args12, terms_J2_1, params)
        e2 = _contract_barebones(contract_o_12, args12, terms_J2_2, params)
        params.verbosity >= 4 && println("bond_J2H = $(J2 * (e1 + e2)/n12)")
        etol += J2 * (e1 + e2)/n12
        e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2 * e1/n12
        e_dict["bond_J2H2_energy"]["$(i),$(j)"] = J2 * e2/n12

        # Vertical inter-cell bonds:
        # J1V: 2(cell) - 1(down); J2V: same-sublattice bonds on 1 and 2.
        ir = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        args21 = (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j],
                  FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j])
        n21 = _contract_one(contract_n_21, args21, params)

        e = _contract_barebones(contract_o_21, args21, terms_J1_inter, params)
        params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n21)")
        etol += J1v * e/n21
        e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n21

        e1 = _contract_barebones(contract_o_21, args21, terms_J2_1, params)
        e2 = _contract_barebones(contract_o_21, args21, terms_J2_2, params)
        params.verbosity >= 4 && println("bond_J2V = $(J2 * (e1 + e2)/n21)")
        etol += J2 * (e1 + e2)/n21
        e_dict["bond_J2V_energy"]["$(i),$(j)"] = J2 * e1/n21
        e_dict["bond_J2V2_energy"]["$(i),$(j)"] = J2 * e2/n21

        # Plaquette diagonals: 1(right)-1(down) and 2(right)-2(down).
        ir = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        args22 = (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j],
                  FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                  A[i,j], A[i,jr], A[ir,j], A[ir,jr])
        e1 = _contract_barebones(contract_o_22_2, args22, terms_J2_1, params)
        e2 = _contract_barebones(contract_o_22_2, args22, terms_J2_2, params)
        n = _contract_one(contract_n_22, args22, params)
        params.verbosity >= 4 && println("bond_J2/ = $(J2 * (e1 + e2)/n)")
        etol += J2 * (e1 + e2)/n
        e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["bond_J2/2_energy"]["$(i),$(j)"] = J2 * e2/n
    end

    energy_per_site = etol / (2 * len)
    params.verbosity >= 3 && println("energy per site = $energy_per_site")
    return energy_per_site, e_dict
end

function energy_value(model::J1J2{Square}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2 = model
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
        "bond_J2\\_energy"  => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}()
    )

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        J1h, J1v = enlarge_coupling(model, i, j)

        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), terms, params)
        n = _contract_one(contract_n_21, (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]), params)
        params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
        etol += J1v * e/n
        e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

        # J2 diagonal bonds connect same sublattice → no rotation
        _terms_j2 = model.ifrotate ? terms_norot : terms
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), _terms_j2, params)
        e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), _terms_j2, params)
        n = _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e1/n)")
        params.verbosity >= 4 && println("bond_J2/ = $(J2 * e2/n)")
        etol += J2 * (e1/n + e2/n)
        e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2 * e2/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end


function energy_value(model::J1J2{Square}, A, env::PlaquetteVUMPSEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for Plaquette VUMPS energy evaluation"))

    @unpack AL, C, FLu, FLo = env
    @unpack J1, J2 = model
    # Slice2D (grid set ⟺ block obs env, gated to J1J2{Square} in ObsEnv): contract the energy
    # expectation block-distributed via the slice2d transfer maps (grid threaded into oc_*),
    # instead of gathering the full env. AC needs the slice2d AL·C (gather AL → serial → scatter).
    grid = _effective_grid(params.boundary_alg)
    AC = grid === nothing ? ALCtoAC(AL, C) : ALCtoAC_slice2d(AL, C, grid)
    Ni, Nj = size(A)
    atype = _arraytype(A[1])

    etol = 0.0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
        "bond_J2\\_energy" => Dict{String, Any}()
    )

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        J1h, J1v = enlarge_coupling(model, i, j)
        ir = mod1(i + 1, Ni)
        jr = mod1(j + 1, Nj)

        params.verbosity >= 4 && println("===========$i,$j===========")
        args12 = (FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr])
        e = _contract_barebones(contract_o_12, args12, terms, params; grid)
        n = _contract_one(contract_n_12, args12, params; grid)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        args21 = (AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j])
        e = _contract_barebones(contract_o_21, args21, terms, params; grid)
        n = _contract_one(contract_n_21, args21, params; grid)
        params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
        etol += J1v * e/n
        e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

        # J2 diagonal bonds connect same sublattice → no rotation
        _terms_j2 = model.ifrotate ? terms_norot : terms
        args22 = (FLu[i,j], FLo[ir,j], AL[i,j], AL[i,j], FLu[i,j], FLo[ir,j], AC[i,jr], AC[i,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr])
        e = _contract_barebones(contract_o_22_1, args22, _terms_j2, params; grid)
        n = _contract_one(contract_n_22, args22, params; grid)
        params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e/n)")
        etol += J2 * e/n * 2 # factor of 2 for the two diagonals in the plaquette
        e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2 * e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::J1J2{Square}, A, env::C4vVUMPSEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for C4v VUMPS energy evaluation"))

    @unpack AL, C, FL = env
    @unpack J1, J2 = model
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J2\\_energy"  => Dict{String, Any}(),
    )
    etol = 0.0

    A1 = A[1]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)
    AC = ALCtoAC_map(AL,C)

    e = _contract_barebones(contract_o_12, (FL, AL, A1, conj(AL), FL, AC, A1, conj(AC)), terms, params)
    n = _contract_one(contract_n_12, (FL, AL, A1, conj(AL), FL, AC, A1, conj(AC)), params)
    params.verbosity >= 4 && println("bond_J1H = $(J1 * e/n)")
    etol += J1 * e/n
    e_dict["bond_J1H_energy"]["1,1"] = J1 * e/n

    # J2 diagonal bonds connect same sublattice → no rotation
    _terms_j2 = model.ifrotate ? terms_norot : terms
    e = _contract_barebones(contract_o_22_1, (FL, FL, AL, conj(AL), FL, FL, AC, conj(AC), A1, A1, A1, A1), _terms_j2, params)
    n = _contract_one(contract_n_22, (FL, FL, AL, conj(AL), FL, FL, AC, conj(AC), A1, A1, A1, A1), params)
    params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e/n)")
    etol += J2 * e/n
    e_dict["bond_J2\\_energy"]["1,1"] = J2 * e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::J1J2{Square}, A, env::CTMEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for QRCTMRG{C4v} energy evaluation"))

    @unpack C, T = env
    @unpack J1, J2 = model
    etol = 0.0
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J2\\_energy" => Dict{String, Any}(),
    )

    A1 = A[1]
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    To = CTCtoT(C, T)
    e = _contract_barebones(contract_o_12, (To, T, A1, T, To, T, A1, T), terms, params)
    n = _contract_one(contract_n_12, (To, T, A1, T, To, T, A1, T), params)
    params.verbosity >= 4 && println("bond_J1H = $(J1 * e/n)")
    etol += J1 * e/n
    e_dict["bond_J1H_energy"]["1,1"] = J1 * e/n

    # J2 diagonal bonds connect same sublattice → no rotation
    _terms_j2 = model.ifrotate ? terms_norot : terms
    @tensor Tu[1,3,4,5] := C[1,2] * T[2,3,4,5]
    @tensor Td[1,2,3,5] := T[1,2,3,4] * C[4,5]
    e = _contract_barebones(contract_o_22_1, (Tu, Td, T, conj(T), Tu, Td, T, conj(T), A1, A1, A1, A1), _terms_j2, params)
    n = _contract_one(contract_n_22, (Tu, Td, T, conj(T), Tu, Td, T, conj(T), A1, A1, A1, A1), params)
    params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e/n)")
    etol += J2 * e/n
    e_dict["bond_J2\\_energy"]["1,1"] = J2 * e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::J1J2{Honeycomb{:brickwall_h}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2 = model
    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    len = length(ACu.data)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J2H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
        "bond_J2\\_energy"  => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}()
    )

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)
        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms, params)
            n = _contract_one(contract_n_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), params)
            params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
            etol += J1v * e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        # J2 diagonal bonds connect same sublattice → no rotation
        _terms_j2 = model.ifrotate ? terms_norot : terms
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), _terms_j2, params)
        e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), _terms_j2, params)
        n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e1/n)")
        params.verbosity >= 4 && println("bond_J2/ = $(J2 * e2/n)")
        etol += J2 * (e1 + e2)/n
        e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2 * e2/n

        # J2 horizontal next-nearest-neighbor bond (skip one column)
        _terms_j2h = model.ifrotate ? terms_norot : terms
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        jrr = mod1(j + 2, Nj)
        e = _contract_barebones(contract_o_13, (FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), _terms_j2h, params)
        n = _contract_one(contract_n_13, (FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), params)
        params.verbosity >= 4 && println("bond_J2H = $(J2 * e/n)")
        etol += J2 * e/n
        e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2 * e/n
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::J1J2{Honeycomb{:brickwall_h}}, A, env::PlaquetteVUMPSEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for Plaquette VUMPS energy evaluation"))

    @unpack AL, C, FLu, FLo = env
    @unpack J2 = model
    AC = ALCtoAC(AL, C)

    atype = _arraytype(A[1])
    Ni, Nj = size(A)
    len = length(A.data)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J2H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
        "bond_J2\\_energy"  => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}()
    )

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)
        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jo = mod1(Nj - j, Nj)
            e = _contract_barebones(contract_o_21, (AC[i,j],FLu[i,j],A[i,j],FLu[i,jo],FLo[ir,j],A[ir,j],FLo[ir,jo],AC[irr,j]), terms, params)
            n = _contract_one(contract_n_21, (AC[i,j],FLu[i,j],A[i,j],FLu[i,jo],FLo[ir,j],A[ir,j],FLo[ir,jo],AC[irr,j]), params)
            params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
            etol += J1v * e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        jo = mod1(Nj - (j+1), Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j],AL[i,j],A[i,j],AL[ir,j],FLo[i,jo],AC[i,jr],A[i,jr],AC[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j],AL[i,j],A[i,j],AL[ir,j],FLo[i,jo],AC[i,jr],A[i,jr],AC[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        # J2 diagonal bonds connect same sublattice → no rotation
        _terms_j2 = model.ifrotate ? terms_norot : terms
        ir  = mod1(i + 1, Ni)
        id = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        jo = mod1(Nj - (j+1), Nj)
        e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AC[i,jr], AC[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), _terms_j2, params)
        e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AC[i,jr], AC[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), _terms_j2, params)
        n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], AL[i,j], AL[id,j], FLu[i,jo], FLo[ir,jo], AC[i,jr], AC[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J2\\ = $(J2 * e1/n)")
        params.verbosity >= 4 && println("bond_J2/ = $(J2 * e2/n)")
        etol += J2 * (e1 + e2)/n
        e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2 * e2/n

        # J2 horizontal next-nearest-neighbor bond (skip one column)
        _terms_j2h = model.ifrotate ? terms_norot : terms
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        jrr = mod1(j + 2, Nj)
        jo = mod1(Nj - (j+2), Nj)
        e = _contract_barebones(contract_o_13, (FLo[i,j], AL[i,j], AL[ir,j], FLo[i,jo], AL[i,jr], AL[ir,jr], AC[i,jrr], AC[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), _terms_j2h, params)
        n = _contract_one(contract_n_13, (FLo[i,j], AL[i,j], AL[ir,j], FLo[i,jo], AL[i,jr], AL[ir,jr], AC[i,jrr], AC[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), params)
        params.verbosity >= 4 && println("bond_J2H = $(J2 * e/n)")
        etol += J2 * e/n
        e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2 * e/n
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end
