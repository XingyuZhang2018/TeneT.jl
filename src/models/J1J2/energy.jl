function energy_value(model::J1J2{Square}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2 = model
    @unpack forloop_iter, ifcheckpoint = params
    @unpack ifparallel = params.boundary_alg

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "J1_Horizontal_energy" => Dict{String, Any}(),
        "J1_Vertical_energy"   => Dict{String, Any}(),
        "J2_Diagonal\\_energy"  => Dict{String, Any}(),
        "J2_Diagonal/_energy"  => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        J1h, J1v = enlarge_coupling(model, i, j)
        O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))

        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J1_Horizontal_energy = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["J1_Horizontal_energy"]["$(i),$(j)"] = J1h * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = contract_o_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; ifparallel, forloop_iter)
        n = contract_n_21(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J1_Vertical_energy = $(J1v * e/n)")
        etol += J1v * e/n
        e_dict["J1_Vertical_energy"]["$(i),$(j)"] = J1v * e/n

        if model.ifrotate
            model.ifrotate = false
            O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
            model.ifrotate = true
        end
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = contract_o_22_1(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; ifparallel, forloop_iter)
        e2 = contract_o_22_2(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; ifparallel, forloop_iter)
        n = contract_n_22(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J2_Diagonal\\_energy = $(J2 * e1/n)")
        params.verbosity >= 4 && println("J2_Diagonal/_energy = $(J2 * e2/n)")
        etol += J2 * (e1/n + e2/n)
        e_dict["J2_Diagonal\\_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["J2_Diagonal/_energy"]["$(i),$(j)"] = J2 * e2/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end
    

function energy_value(model::J1J2{Square}, A, env::PlaquetteVUMPSEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for Plaquette VUMPS energy evaluation"))

    @unpack AL, C, FLu, FLo = env
    @unpack J1, J2 = model
    AC = ALCtoAC(AL, C)
    Ni, Nj = size(A)
    atype = _arraytype(A[1])

    etol = 0.0
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "J1_Horizontal_energy" => Dict{String, Any}(),
        "J1_Vertical_energy"   => Dict{String, Any}(),
        "J2_Diagonal\\_energy" => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        J1h, J1v = enlarge_coupling(model, i, j)
        ir = mod1(i + 1, Ni)
        jr = mod1(j + 1, Nj)

        O1, O2 = atype.(hamiltonian_trunc(model))

        params.verbosity >= 4 && println("===========$i,$j===========")
        e = contract_o_12(FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr], O1, O2; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J1_Horizontal_energy = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["J1_Horizontal_energy"]["$(i),$(j)"] = J1h * e/n

        e = contract_o_21(AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j], O1, O2; ifparallel, forloop_iter)
        n = contract_n_21(AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J1_Vertical_energy = $(J1v * e/n)")
        etol += J1v * e/n
        e_dict["J1_Vertical_energy"]["$(i),$(j)"] = J1v * e/n

        if model.ifrotate
            model.ifrotate = false
            O1, O2 = atype.(hamiltonian_trunc(model))
            model.ifrotate = true
        end
        e = contract_o_22_1(FLu[i,j], FLo[ir,j], AL[i,j], AL[i,j], FLu[i,j], FLo[ir,j], AC[i,jr], AC[i,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; ifparallel, forloop_iter)
        n = contract_n_22(FLu[i,j], FLo[ir,j], AL[i,j], AL[i,j], FLu[i,j], FLo[ir,j], AC[i,jr], AC[i,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J2_Diagonal\\_energy = $(J2 * e/n)")
        etol += J2 * e/n * 2 # factor of 2 for the two diagonals in the plaquette
        e_dict["J2_Diagonal\\_energy"]["$(i),$(j)"] = J2 * e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::J1J2{Square}, A, env::C4vVUMPSEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for C4v VUMPS energy evaluation"))

    @unpack AL, C, FL = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    @unpack J1, J2 = model
    e_dict = Dict{String, Dict{String, Any}}(
        "J1_Horizontal_energy" => Dict{String, Any}(),
        "J2_Diagonal\\_energy"  => Dict{String, Any}(),
    )
    etol = 0.0

    A1 = A[1]
    atype = _arraytype(A1)
    O1, O2 = atype.(hamiltonian_trunc(model))
    AC = ALCtoAC_map(AL,C)

    e = contract_o_12(FL, AL, A1, conj(AL), FL, AC, A1, conj(AC), O1, O2; ifparallel, forloop_iter)
    n = contract_n_12(FL, AL, A1, conj(AL), FL, AC, A1, conj(AC); ifparallel, forloop_iter)
    params.verbosity >= 4 && println("J1_Horizontal energy = $(J1 * e/n)")
    etol += J1 * e/n
    e_dict["J1_Horizontal_energy"]["1,1"] = J1 * e/n

    if model.ifrotate
        model.ifrotate = false
        O1, O2 = atype.(hamiltonian_trunc(model))
        model.ifrotate = true
    end
    e = contract_o_22_1(FL, FL, AL, conj(AL), FL, FL, AC, conj(AC), A1, A1, A1, A1, O1, O2; ifparallel, forloop_iter)
    n = contract_n_22(FL, FL, AL, conj(AL), FL, FL, AC, conj(AC), A1, A1, A1, A1; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("J2_Diagonal\\_energy= $(J2 * e/n)")
    etol += J2 * e/n
    e_dict["J2_Diagonal\\_energy"]["1,1"] = J2 * e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::J1J2{Square}, A, env::CTMEnv, params::iPEPSOptimize)
    model.ifrotate == true || throw(ArgumentError("model.ifrotate must be true for QRCTM energy evaluation"))

    @unpack C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    @unpack J1, J2 = model
    etol = 0.0
    e_dict = Dict{String, Dict{String, Any}}(
        "J1_Horizontal_energy" => Dict{String, Any}(),
        "J2_Diagonal\\_energy" => Dict{String, Any}(),
    )
    
    A1 = A[1]
    atype = _arraytype(A1)
    O1, O2 = atype.(hamiltonian_trunc(model))

    To = CTCtoT(C, T)
    e = contract_o_12(To, T, A1, T, To, T, A1, T, O1, O2; ifparallel, forloop_iter)
    n = contract_n_12(To, T, A1, T, To, T, A1, T; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("J1_Horizontal_energy = $(J1 * e/n)")
    etol += J1 * e/n
    e_dict["J1_Horizontal_energy"]["1,1"] = J1 * e/n

    if model.ifrotate
        model.ifrotate = false
        O1, O2 = atype.(hamiltonian_trunc(model))
        model.ifrotate = true
    end
    @tensor Tu[1,3,4,5] := C[1,2] * T[2,3,4,5] 
    @tensor Td[1,2,3,5] := T[1,2,3,4] * C[4,5]
    e = contract_o_22_1(Tu, Td, T, conj(T), Tu, Td, T, conj(T), A1, A1, A1, A1, O1, O2; ifparallel, forloop_iter)
    n = contract_n_22(Tu, Td, T, conj(T), Tu, Td, T, conj(T), A1, A1, A1, A1; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("J2_Diagonal\\_energy= $(J2 * e/n)")
    etol += J2 * e/n
    e_dict["J2_Diagonal\\_energy"]["1,1"] = J2 * e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function energy_value(model::J1J2{Honeycomb{:brickwall}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2 = model
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    len = length(ACu.data)

    e_dict = Dict{String, Dict{String, Any}}(
        "J1_Horizontal_energy" => Dict{String, Any}(),
        "J2_Horizontal_energy" => Dict{String, Any}(),
        "J1_Vertical_energy"   => Dict{String, Any}(),
        "J2_Diagonal\_energy"  => Dict{String, Any}(),
        "J2_Diagonal/_energy"  => Dict{String, Any}()
    )
    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
        
        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)
        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni) 
            e = contract_o_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j], O1, O2; ifparallel, forloop_iter)
            n = contract_n_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("J1_Vertical_energy = $(J1v * e/n)")
            etol += J1v * e/n
            e_dict["J1_Vertical_energy"]["$(i),$(j)"] = J1v * e/n
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr], O1, O2; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J1_Horizontal_energy = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["J1_Horizontal_energy"]["$(i),$(j)"] = J1h * e/n

        if model.ifrotate
            model.ifrotate = false
            O1, O2 = atype.(hamiltonian_trunc(model))
            model.ifrotate = true
        end
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = contract_o_22_1(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; ifparallel, forloop_iter)
        e2 = contract_o_22_2(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; ifparallel, forloop_iter)
        n =  contract_n_22(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J2_Diagonal\_energy = $(J2 * e1/n)")
        params.verbosity >= 4 && println("J2_Diagonal/_energy = $(J2 * e2/n)")
        etol += J2 * (e1 + e2)/n
        e_dict["J2_Diagonal\_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["J2_Diagonal/_energy"]["$(i),$(j)"] = J2 * e2/n

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        jrr = mod1(j + 2, Nj)
        e = contract_o_13(FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr], O1, O2; ifparallel, forloop_iter)
        n = contract_n_13(FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("J2_Horizontal_energy = $(J2 * e/n)")
        etol += J2 * e/n
        e_dict["J2_Horizontal_energy"]["$(i),$(j)"] = J2 * e/n
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end