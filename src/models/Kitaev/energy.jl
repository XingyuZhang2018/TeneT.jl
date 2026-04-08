function energy_value(model::Kitaev{Honeycomb{:brickwall}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    atype = _arraytype(A[1])
    Ni, Nj = size(A)

    h = hamiltonian(model)
    Sx1, Sx2 = Zygote.@ignore atype.(hamiltonian_trunc(h[1]))
    Sy1, Sy2 = Zygote.@ignore atype.(hamiltonian_trunc(h[2]))
    Sz1, Sz2 = Zygote.@ignore atype.(hamiltonian_trunc(h[3]))

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_Jx_energy" => Dict{String, Any}(),
        "bond_Jy_energy" => Dict{String, Any}(),
        "bond_Jz_energy" => Dict{String, Any}()
    )
    etol = 0.0

    for (i, j) in eachindex(A)
        params.verbosity >= 4 && println("===========$i,$j===========")
        Jx, Jy, Jz = enlarge_coupling(model, i, j)
        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni) 
            e = contract_o_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j], Sy1, Sy2; ifparallel, forloop_iter)
            n = contract_n_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_Jy = $(Jy * e/n)")
            etol += Jy * e/n
            e_dict["bond_Jy_energy"]["$(i),$(j)"] = Jy * e/n

            O_H = (Sx1, Sx2)
        else
            O_H = (Sz1, Sz2)
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr], O_H[1], O_H[2]; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]; ifparallel, forloop_iter)
        if (i + j) % 2 != 0 
            params.verbosity >= 4 && println("bond_Jx = $(Jx * e/n)") 
            etol += Jx * e/n
            e_dict["bond_Jx_energy"]["$(i),$(j)"] = Jx * e/n
        else
            params.verbosity >= 4 && println("bond_Jz = $(Jz * e/n)")
            etol += Jz * e/n
            e_dict["bond_Jz_energy"]["$(i),$(j)"] = Jz * e/n
        end
    end

    esite = etol/length(A)
    params.verbosity >= 3 && println("energy = $(esite)")
    return esite, e_dict
end