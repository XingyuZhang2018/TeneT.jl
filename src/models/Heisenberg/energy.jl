function energy_value(model::Heisenberg{Square}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = atype.(hamiltonian_trunc(model))
    etol = 0
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
        "Vertical_energy"   => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter, ifparallel)
        n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter, ifparallel)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = contract_o_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter, ifparallel)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter, ifparallel)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = e/n
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
    O1, O2 = atype.(hamiltonian_trunc(model))
    etol = 0
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
        "Vertical_energy"   => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        ir = mod1(i + 1, Ni)
        jr = mod1(j + 1, Nj)

        params.verbosity >= 4 && println("===========$i,$j===========")
        e = contract_o_12(FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr], O1, O2; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = e/n

        e = contract_o_21(AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j], O1, O2; ifparallel, forloop_iter)
        n = contract_n_21(AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::C4vVUMPSEnv, params::iPEPSOptimize)
    @unpack AL, C, FL = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
    )

    A1 = A[1]
    O1, O2 = _arraytype(A1).(hamiltonian_trunc(model))
    AC = ALCtoAC_map(AL,C)

    e = contract_o_12(FL, AL, A1, conj(AL), FL, AC, A1, conj(AC), O1, O2; ifparallel, forloop_iter)
    n = contract_n_12(FL, AL, A1, conj(AL), FL, AC, A1, conj(AC); ifparallel, forloop_iter)
    params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
    etol = e/n
    e_dict["Horizontal_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::CTMEnv, params::iPEPSOptimize)
    @unpack C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
    )

    # Extract site tensor once to avoid repeated StructArray indexing in AD
    A1 = A[1]
    O1, O2 = _arraytype(A1).(hamiltonian_trunc(model))

    To = CTCtoT(C, T)
    e = contract_o_12(To, T, A1, T, To, T, A1, T, O1, O2; ifparallel, forloop_iter)
    n = contract_n_12(To, T, A1, T, To, T, A1, T; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
    etol = e/n
    e_dict["Horizontal_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

"""
    energy_value_perbond(model::Heisenberg{Kagome{:merge}}, A, env, params)

Compute per-bond energies for Kagome merge (6 bonds per unit cell).
More expensive than energy_value — use only for observation, not optimization.

Returns `e_dict` with keys:
- `"bond_12_energy"`: intra-cell, site 1 ↔ site 2
- `"bond_13_energy"`: intra-cell, site 1 ↔ site 3
- `"bond_23_energy"`: intra-cell, site 2 ↔ site 3
- `"bond_31H_energy"`: inter-cell horizontal, site 3@(i,j) ↔ site 1@(i,j+1)
- `"bond_32H_energy"`: inter-cell horizontal, site 3@(i,j) ↔ site 2@(i,j+1)
- `"bond_31V_energy"`: inter-cell vertical, site 3@(i,j) ↔ site 1@(i+1,j)
- `"bond_21V_energy"`: inter-cell vertical, site 2@(i,j) ↔ site 1@(i+1,j)
"""
function energy_value_perbond(model::Heisenberg{Kagome{:merge}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    @unpack Jx, Jy, Jz, S = model

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    len = length(A)

    Sx = const_Sx(S); Sy = const_Sy(S); Sz = const_Sz(S)
    d = size(Sx, 1)
    Id = Matrix{Float64}(I, d, d)

    # Intra-cell onsite operators (d³ × d³)
    h_12 = Jx * reshape((@tensor o[1,2,3,4,5,6] := Sx[1,4] * Sx[2,5] * Id[3,6]), d^3, d^3) +
           Jy * reshape((@tensor o[1,2,3,4,5,6] := Sy[1,4] * Sy[2,5] * Id[3,6]), d^3, d^3) +
           Jz * reshape((@tensor o[1,2,3,4,5,6] := Sz[1,4] * Sz[2,5] * Id[3,6]), d^3, d^3)
    h_13 = Jx * reshape((@tensor o[1,2,3,4,5,6] := Sx[1,4] * Id[2,5] * Sx[3,6]), d^3, d^3) +
           Jy * reshape((@tensor o[1,2,3,4,5,6] := Sy[1,4] * Id[2,5] * Sy[3,6]), d^3, d^3) +
           Jz * reshape((@tensor o[1,2,3,4,5,6] := Sz[1,4] * Id[2,5] * Sz[3,6]), d^3, d^3)
    h_23 = Jx * reshape((@tensor o[1,2,3,4,5,6] := Id[1,4] * Sx[2,5] * Sx[3,6]), d^3, d^3) +
           Jy * reshape((@tensor o[1,2,3,4,5,6] := Id[1,4] * Sy[2,5] * Sy[3,6]), d^3, d^3) +
           Jz * reshape((@tensor o[1,2,3,4,5,6] := Id[1,4] * Sz[2,5] * Sz[3,6]), d^3, d^3)
    h_12 = atype(real(h_12))
    h_13 = atype(real(h_13))
    h_23 = atype(real(h_23))

    # Inter-cell horizontal operators: site 3@left ↔ site 1@right, site 3@left ↔ site 2@right
    h_31H = Jx * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sx[3,9] * Sx[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jy * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sy[3,9] * Sy[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jz * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sz[3,9] * Sz[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3)
    h_32H = Jx * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sx[3,9] * Id[4,10]*Sx[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jy * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sy[3,9] * Id[4,10]*Sy[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jz * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sz[3,9] * Id[4,10]*Sz[5,11]*Id[6,12]), d^3,d^3,d^3,d^3)
    Oh31_1, Oh31_2 = atype.(hamiltonian_trunc(real(h_31H)))
    Oh32_1, Oh32_2 = atype.(hamiltonian_trunc(real(h_32H)))

    # Inter-cell vertical operators: site 3@upper ↔ site 1@lower, site 2@upper ↔ site 1@lower
    h_31V = Jx * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sx[3,9] * Sx[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jy * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sy[3,9] * Sy[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jz * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Id[2,8]*Sz[3,9] * Sz[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3)
    h_21V = Jx * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Sx[2,8]*Id[3,9] * Sx[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jy * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Sy[2,8]*Id[3,9] * Sy[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3) +
            Jz * reshape((@tensor o[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7]*Sz[2,8]*Id[3,9] * Sz[4,10]*Id[5,11]*Id[6,12]), d^3,d^3,d^3,d^3)
    Ov31_1, Ov31_2 = atype.(hamiltonian_trunc(real(h_31V)))
    Ov21_1, Ov21_2 = atype.(hamiltonian_trunc(real(h_21V)))

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_12_energy"  => Dict{String, Any}(),
        "bond_13_energy"  => Dict{String, Any}(),
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

        e = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], h_13; ifparallel, forloop_iter)
        e_dict["bond_13_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_13 = $(e/n)")

        e = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], h_23; ifparallel, forloop_iter)
        e_dict["bond_23_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_23 = $(e/n)")

        # Inter-cell horizontal bonds
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; ifparallel, forloop_iter)

        e = contract_o_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], Oh31_1, Oh31_2; ifparallel, forloop_iter)
        e_dict["bond_31H_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_31H = $(e/n)")

        e = contract_o_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], Oh32_1, Oh32_2; ifparallel, forloop_iter)
        e_dict["bond_32H_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_32H = $(e/n)")

        # Inter-cell vertical bonds
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; ifparallel, forloop_iter)

        e = contract_o_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], Ov31_1, Ov31_2; ifparallel, forloop_iter)
        e_dict["bond_31V_energy"]["$(i),$(j)"] = e / n
        params.verbosity >= 4 && println("bond_31V = $(e/n)")

        e = contract_o_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], Ov21_1, Ov21_2; ifparallel, forloop_iter)
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
        "onsite_energy"   => Dict{String, Any}(),
        "Horizontal_energy" => Dict{String, Any}(),
        "Vertical_energy"   => Dict{String, Any}()
    )

    h_H, h_V = hamiltonian(model)
    Oh1, Oh2 = atype.(hamiltonian_trunc(h_H))
    Ov1, Ov2 = atype.(hamiltonian_trunc(h_V))
    h_onsite = atype(hamiltonian_onsite(model))
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")

        ir = Ni + 1 - i
        e = contract_o_11(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j], h_onsite; ifparallel, forloop_iter)
        n = contract_n_11(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Onsite energy = $(e/n)")
        etol += e/n
        e_dict["onsite_energy"]["$(i),$(j)"] = e/n

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], Oh1, Oh2; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = contract_o_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], Ov1, Ov2; ifparallel, forloop_iter)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end