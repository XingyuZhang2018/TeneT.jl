function energy_value(model::Heisenberg{Square}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
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
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter, ifparallel)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter, ifparallel)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter, ifparallel)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter, ifparallel)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function energy_value(model::Heisenberg{Square}, A, env::PlaquetteVUMPSEnv, params::iPEPSOptimize)
    @unpack AL, C, FLu, FLo = env
    AC = ALCtoAC(AL, C)
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
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
        e = contract_o2_H(FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr], O1, O2; ifparallel, forloop_iter)
        n = contract_n2_H(FLo[i,j], AL[i,j], A[i,j], AL[ir,j], FLo[i,j], AC[i,jr], A[i,jr], AC[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = e/n

        e = contract_o2_V(AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j], O1, O2; ifparallel, forloop_iter)
        n = contract_n2_V(AC[i,j], FLu[i,j], A[i,j], FLu[i,jr], FLo[ir,j], A[ir,j], FLo[ir,jr], AC[i,j]; ifparallel, forloop_iter)
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

    O1, O2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))
    AC = ALCtoAC_map(AL,C)

    e = contract_o2_H(FL, AL, A[1], conj(AL), FL, AC, A[1], conj(AC), O1, O2; ifparallel, forloop_iter)
    n = contract_n2_H(FL, AL, A[1], conj(AL), FL, AC, A[1], conj(AC); ifparallel, forloop_iter)
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

    O1, O2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))

    To = CTCtoT(C, T)
    e = contract_o2_H(To, T, A[1], T, To, T, A[1], T, O1, O2; ifparallel, forloop_iter)
    n = contract_n2_H(To, T, A[1], T, To, T, A[1], T; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
    etol = e/n
    e_dict["Horizontal_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end
