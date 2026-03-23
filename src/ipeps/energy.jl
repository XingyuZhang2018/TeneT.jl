# Energy computation for iPEPS optimization
# Dispatches on model type and environment type (VUMPSEnv / CTMEnv)

"""
    energy(A, env, params::iPEPSOptimize)

Main entry point for energy computation during iPEPS optimization.
Calls `expectation_value` dispatching on `params.model` type.
"""
function energy(A, env, params::iPEPSOptimize)
    return expectation_value(params.model, A, env, params)[1]
end

# ============================================================================
# Heisenberg model
# ============================================================================

function expectation_value(model::Heisenberg, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
    etol = 0
    forloop_iter = params.forloop_iter
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
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function expectation_value(model::Heisenberg, A, env::CTMEnv, params::iPEPSOptimize)
    @unpack C, T = env
    @unpack forloop_iter = params

    h1, h2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))
    D, d = size(A)[[1, 5]]
    Dh = size(h1, 3)

    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
    )

    A1u = reshape((@tensor A1u[a,b,c,i,d,f] := A[a,b,c,d,e] * h1[e,f,i]), D,D,D*Dh,D,d)
    A2u = reshape((@tensor A2u[a,i,b,c,d,f] := A[a,b,c,d,e] * h2[i,e,f]), D*Dh,D,D,D,d)

    To = CTCtoT(C, T)
    e = oc_H_leg4(To, T, T, A1u, A, A2u, A; forloop_iter)
    n = oc_H_leg4(To, T, T, A, A, A, A; forloop_iter)
    params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
    etol = e/n
    e_dict["Horizontal_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

# ============================================================================
# J1J2 model
# ============================================================================

function expectation_value(model::J1J2, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2 = model
    @unpack forloop_iter = params

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
        "Vertical_energy"   => Dict{String, Any}(),
        "Diagonal1_energy"  => Dict{String, Any}(),
        "Diagonal2_energy"  => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))

        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += J1 * e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = J1 * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += J1 * e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = J1 * e/n

        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = contract_o_D1(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
        e2 = contract_o_D2(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
        n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("h2D1 = $(J2*e1/n)")
        params.verbosity >= 4 && println("h2D2 = $(J2*e2/n)")
        etol += J2 * (e1/n + e2/n)
        e_dict["Diagonal1_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["Diagonal2_energy"]["$(i),$(j)"] = J2 * e2/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

function expectation_value(model::J1J2, A, env::CTMEnv, params::iPEPSOptimize)
    @unpack C, T = env
    @unpack forloop_iter = params
    @unpack J1, J2 = model

    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
        "Diagonal1_energy"  => Dict{String, Any}()
    )

    h1, h2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))
    D, d = size(A)[[1, 5]]
    Dh = size(h1, 3)

    A1u = reshape((@tensor A1u[a,b,c,i,d,f] := A[a,b,c,d,e] * h1[e,f,i]), D,D,D*Dh,D,d)
    A2u = reshape((@tensor A2u[a,i,b,c,d,f] := A[a,b,c,d,e] * h2[i,e,f]), D*Dh,D,D,D,d)

    To = CTCtoT(C, T)
    e = oc_H_leg4(To, T, T, A1u, A, A2u, A; forloop_iter)
    n = oc_H_leg4(To, T, T, A, A, A, A; forloop_iter)
    params.verbosity >= 4 && println("Horizontal energy = $(J1 * e/n)")
    etol = J1 * e/n
    e_dict["Horizontal_energy"]["1,1"] = J1 * e/n

    # Diagonal term
    h1, h2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))
    @tensor Tu[1,3,4,5] := C[1,2] * T[2,3,4,5]
    @tensor Td[1,2,3,5] := T[1,2,3,4] * C[4,5]
    e = contract_o_D1(Tu, Td, T, T, Tu, Td, T, T, A, h1, h2; forloop_iter)
    n = contract_n_D(Tu, Td, T, T, Tu, Td, T, T, A; forloop_iter)
    params.verbosity >= 4 && println("h2D1 = $(J2 * e/n)")
    etol += J2 * e/n
    e_dict["Diagonal1_energy"]["1,1"] = J2 * e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

# ============================================================================
# J1J2J3 model
# ============================================================================

function expectation_value(model::J1J2J3, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2, J3 = model
    @unpack forloop_iter = params

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "J1_Horizontal_energy" => Dict{String, Any}(),
        "J1_Vertical_energy"   => Dict{String, Any}(),
        "J3_Horizontal_energy" => Dict{String, Any}(),
        "J3_Vertical_energy"   => Dict{String, Any}(),
        "J2_Diagonal1_energy"  => Dict{String, Any}(),
        "J2_Diagonal2_energy"  => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))

        params.verbosity >= 4 && println("===========$i,$j===========")
        id = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[id,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[id,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[id,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[id,jr]; forloop_iter)
        params.verbosity >= 4 && println("J1_Horizontal_energy = $(J1*e/n)")
        etol += J1 * e/n
        e_dict["J1_Horizontal_energy"]["$(i),$(j)"] = J1 * e/n

        ir  =  mod1(i + 1, Ni)
        id = mod1(Ni - i, Ni)
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[id,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[id,j]; forloop_iter)
        params.verbosity >= 4 && println("J1_Vertical energy = $(J1*e/n)")
        etol += J1 * e/n
        e_dict["J1_Vertical_energy"]["$(i),$(j)"] = J1 * e/n

        if (i + j) % 2 == 0
            ir  = mod1(i + 1, Ni)
            id = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e = contract_o_D1(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            params.verbosity >= 4 && println("J2_Diagonal1_energy = $(J2*e/n)")
            etol += J2 * e/n
            e_dict["J2_Diagonal1_energy"]["$(i),$(j)"] = J2 * e/n

            id = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            jrr = mod1(j + 2, Nj)
            e = contract_o3_H(FLo[i,j], ACu[i,j], ACd[id,j], FRo[i,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], O1, O2; forloop_iter)
            n = contract_n3_H(FLo[i,j], ACu[i,j], ACd[id,j], FRo[i,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr]; forloop_iter)
            params.verbosity >= 4 && println("J3_Horizontal_energy = $(J3*e/n)")
            etol += J3 * e/n
            e_dict["J3_Horizontal_energy"]["$(i),$(j)"] = J3 * e/n

            ir = mod1(i + 1, Ni)
            irr = mod1(i + 2, Ni)
            id = mod1(Ni - i - 1, Ni)
            e = contract_o3_V(ACu[i,j], ACd[id,j], FLu[i,j], FRu[i,j], FLu[ir,j], FRu[ir,j], FLo[irr,j], FRo[irr,j], A[i,j], A[ir,j], A[irr,j], O1, O2; forloop_iter)
            n = contract_n3_V(ACu[i,j], ACd[id,j], FLu[i,j], FRu[i,j], FLu[ir,j], FRu[ir,j], FLo[irr,j], FRo[irr,j], A[i,j], A[ir,j], A[irr,j]; forloop_iter)
            params.verbosity >= 4 && println("J3_Vertical_energy = $(J3*e/n)")
            etol += J3 * e/n
            e_dict["J3_Vertical_energy"]["$(i),$(j)"] = J3 * e/n
        else
            ir  = mod1(i + 1, Ni)
            id = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e = contract_o_D2(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[id,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            params.verbosity >= 4 && println("J2_Diagonal2_energy = $(J2*e/n)")
            etol += J2 * e/n
            e_dict["J2_Diagonal2_energy"]["$(i),$(j)"] = J2 * e/n
        end
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end

# ============================================================================
# Shastry-Sutherland (SS) model
# ============================================================================

function expectation_value(model::SS, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2 = model
    @unpack forloop_iter = params

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
    etol = 0
    len = length(A)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += J1 * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += J1 * e/n

        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        if i % 2 == 1 && j % 2 == 0
            n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            e1 = contract_o_D1(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            params.verbosity >= 4 && println("h2D1 = $(J2*e1/n)")
            etol += J2 * e1/n
        elseif i % 2 == 0 && j % 2 == 1
            n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            e2 = contract_o_D2(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            params.verbosity >= 4 && println("h2D2 = $(J2*e2/n)")
            etol += J2 * e2/n
        end
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len
end

# ============================================================================
# Kagome model
# ============================================================================

function expectation_value(model::Kagome, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2 = model
    @unpack forloop_iter = params

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "onsite_energy"     => Dict{String, Any}(),
        "Horizontal_energy" => Dict{String, Any}(),
        "Vertical_energy"   => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        Oh1, Oh2 = Zygote.@ignore atype.(hamiltonian_trunc(model, "right"))
        Ov1, Ov2 = Zygote.@ignore atype.(hamiltonian_trunc(model, "down"))
        h = Zygote.@ignore atype(hamiltonian_onsite(model))

        params.verbosity >= 4 && println("===========$i,$j===========")

        ir = Ni + 1 - i
        e = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], h; forloop_iter)
        n = contract_n1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]; forloop_iter)
        params.verbosity >= 4 && println("Onsite energy = $(e/n)")
        etol += J1 * e/n
        e_dict["onsite_energy"]["$(i),$(j)"] = J1 * e/n

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], Oh1, Oh2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += J1 * e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = J1 * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], Ov1, Ov2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += J1 * e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = J1 * e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    return etol/len, e_dict
end
