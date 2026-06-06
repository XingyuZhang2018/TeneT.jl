# ============================================================================
# Magnetization — generic (Heisenberg, J1J2, J1J2J3, SS)
# ============================================================================

function magnetization_value(model, A, env::VUMPSEnv, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    etype = eltype(ACu[1])
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Ni, Nj = size(ACu)
    len = length(ACu.data)
    @unpack forloop_iter = params
    @unpack  ifparallel = params.boundary_alg
    m_dict = Dict{String, Any}()
    Mnorm = zeros(Float64, Ni, Nj)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        Mx = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx; forloop_iter, ifparallel)
        My = etype <: Real ? 0.0 : contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy; forloop_iter, ifparallel)
        Mz = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz; forloop_iter, ifparallel)

        n = contract_n_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]; forloop_iter, ifparallel)
        Mag = [Mx/n, My/n, Mz/n]
        Mnorm[i,j] = norm(Mag)
        params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")

        m_dict["$(i),$(j)"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm[i,j])
    end

    M_mean = sum(Mnorm)/len
    params.verbosity >= 4 && println("|M|_mean = $(M_mean)")
    return M_mean, m_dict
end

function magnetization_value(model, A, env::PlaquetteVUMPSEnv, params)
    @unpack AL, C, FLu, FLo = env
    AC = ALCtoAC(AL, C)
    atype = _arraytype(AC[1])
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Ni, Nj = size(AC)
    len = length(AC.data)
    Ni,Nj = size(AC)
    ifparallel = params.boundary_alg.ifparallel
    forloop_iter = params.forloop_iter
    m_dict = Dict{String, Any}()
    etype = eltype(AC[1,1])
    Mnorm = zeros(etype, Ni, Nj)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), AC.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = model.lattice isa Square ? mod1(j + 1, Nj) : mod1(Nj - j, Nj)
        Mx = contract_o_11(FLo[i,j],AC[i,j],A[i,j],AC[ir,j],FLo[i,jr], Sx; ifparallel, forloop_iter)
        My = etype == Float64 ? 0.0 : contract_o_11(FLo[i,j],AC[i,j],A[i,j],AC[ir,j],FLo[i,jr], Sy; ifparallel, forloop_iter)
        Mz = contract_o_11(FLo[i,j],AC[i,j],A[i,j],AC[ir,j],FLo[i,jr], Sz; ifparallel, forloop_iter)

        n = contract_n_11(FLo[i,j],AC[i,j],A[i,j],AC[ir,j],FLo[i,jr]; ifparallel, forloop_iter)
        Mag = [Mx/n, My/n, Mz/n]
        Mnorm[i,j] = norm(Mag)
        params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")

        m_dict["$(i),$(j)"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm[i,j])
    end

    M_mean = sum(Mnorm)/len
    params.verbosity >= 4 && println("|M|_mean = $(M_mean)")
    return M_mean, m_dict
end

function magnetization_value(model, A, env::OnesideVUMPSEnv, params)
    @unpack AC, AR, FLu, FRu, FLo, FRo = env
    atype = _arraytype(AC[1])
    etype = eltype(AC[1])
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Ni, Nj = size(AC)
    len = length(AC.data)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    m_dict = Dict{String, Any}()
    Mnorm = zeros(Float64, Ni, Nj)
    ir_oneside(i) = obs_index(typeof(model), i, Ni)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), AC.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = ir_oneside(i)
        Mx = contract_o_11(FLo[i,j], AC[i,j], A[i,j], AC[ir,j], FRo[i,j], Sx; forloop_iter, ifparallel)
        My = etype <: Real ? 0.0 : contract_o_11(FLo[i,j], AC[i,j], A[i,j], AC[ir,j], FRo[i,j], Sy; forloop_iter, ifparallel)
        Mz = contract_o_11(FLo[i,j], AC[i,j], A[i,j], AC[ir,j], FRo[i,j], Sz; forloop_iter, ifparallel)

        n = contract_n_11(FLo[i,j], AC[i,j], A[i,j], AC[ir,j], FRo[i,j]; forloop_iter, ifparallel)
        Mag = [Mx/n, My/n, Mz/n]
        Mnorm[i,j] = norm(Mag)
        params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")

        m_dict["$(i),$(j)"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm[i,j])
    end

    M_mean = sum(Mnorm)/len
    params.verbosity >= 4 && println("|M|_mean = $(M_mean)")
    return M_mean, m_dict
end

function magnetization_value(model, A, env::C4vVUMPSEnv, params)
    @unpack AL, C, FL = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    m_dict = Dict{String, Any}()
    AC = ALCtoAC_map(AL, C)
    atype = _arraytype(AC)
    etype = eltype(AC)
    A1 = A[1]

    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Mx = contract_o_11(FL,AC,A1,AC,FL, Sx; ifparallel, forloop_iter)
    My = etype == Float64 ? 0.0 : contract_o_11(FL,AC,A1,AC,FL, Sy; ifparallel, forloop_iter)
    Mz =contract_o_11(FL,AC,A1,AC,FL, Sz; ifparallel, forloop_iter)

    n = contract_n_11(FL,AC,A1,AC,FL; ifparallel, forloop_iter)
    Mag = [Mx/n, My/n, Mz/n]
    Mnorm = norm(Mag)
    params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")
    m_dict["1,1"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm)

    return Mnorm, m_dict
end

function magnetization_value(model, A, env::CTMEnv, params)
    @unpack C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    m_dict = Dict{String, Any}()
    To = CTCtoT(C, T)
    atype = _arraytype(To)
    etype = eltype(To)
    A1 = A[1]

    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Mx = contract_o_11(To,T,A1,T,To, Sx; ifparallel, forloop_iter)
    My = etype == Float64 ? 0.0 : contract_o_11(To,T,A1,T,To, Sy; ifparallel, forloop_iter)
    Mz =contract_o_11(To,T,A1,T,To, Sz; ifparallel, forloop_iter)

    n = contract_n_11(To,T,A1,T,To; ifparallel, forloop_iter)
    Mag = [Mx/n, My/n, Mz/n]
    Mnorm = norm(Mag)
    params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")
    m_dict["1,1"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm)

    return Mnorm, m_dict
end

function _contract_c3v_one_site(C, T, A, O; ifparallel=false, forloop_iter=1)
    @tensor AO[a,b,c,f] := A[a,b,c,e] * O[e,f]
    Ac = conj(A)
    @tensor result[] := C[x1,x2] * T[x2,a,aa,x3] *
                        C[x3,x4] * T[x4,b,bb,x5] *
                        AO[a,b,c,p] * Ac[aa,bb,cc,p] *
                        C[x5,x6] * T[x6,c,cc,x1]
    return only(result)
end

function _contract_c3v_one_site_norm(C, T, A; ifparallel=false, forloop_iter=1)
    Ac = conj(A)
    @tensor result[] := C[x1,x2] * T[x2,a,aa,x3] *
                        C[x3,x4] * T[x4,b,bb,x5] *
                        A[a,b,c,p] * Ac[aa,bb,cc,p] *
                        C[x5,x6] * T[x6,c,cc,x1]
    return only(result)
end

function magnetization_value(model::Heisenberg{Honeycomb{:c3v}}, A, env::CTMEnv, params)
    @unpack C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    A1 = A[1]
    atype = _arraytype(A1)
    etype = eltype(A1)

    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    n = _contract_one(_contract_c3v_one_site_norm, (C, T, A1), params)
    Mx = checkpoint(params.bond_checkpoint, _contract_c3v_one_site, C, T, A1, Sx; ifparallel, forloop_iter) / n
    My = etype <: Real ? 0.0 :
         checkpoint(params.bond_checkpoint, _contract_c3v_one_site, C, T, A1, Sy; ifparallel, forloop_iter) / n
    Mz = checkpoint(params.bond_checkpoint, _contract_c3v_one_site, C, T, A1, Sz; ifparallel, forloop_iter) / n

    Mag = [Mx, My, Mz]
    Mnorm = norm(Mag)
    params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")

    m_dict = Dict{String, Any}("1,1" => Dict("Mx" => Mx, "My" => My, "Mz" => Mz, "|M|" => Mnorm))
    return Mnorm, m_dict
end

# ============================================================================
# Magnetization — Kagome (3 sublattice sites per unit cell)
# ============================================================================

function magnetization_value(model::Heisenberg{<:KagomeOnehole}, A, env::VUMPSEnv, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    etype = eltype(ACu[1])
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Ni, Nj = size(ACu)
    len = length(ACu.data)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    m_dict = Dict{String, Any}()
    Mnorm = zeros(Float64, Ni, Nj)
    nphys = 0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        if (i % 2 == 0) && (j % 2 == 0)  # empty site: trivial entry, do not contribute to mean
            m_dict["$(i),$(j)"] = Dict("Mx" => 0.0, "My" => 0.0, "Mz" => 0.0, "|M|" => 0.0)
            continue
        end
        nphys += 1
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        Mx = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx; forloop_iter, ifparallel)
        My = etype <: Real ? 0.0 : contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy; forloop_iter, ifparallel)
        Mz = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz; forloop_iter, ifparallel)

        n = contract_n_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]; forloop_iter, ifparallel)
        Mag = [Mx/n, My/n, Mz/n]
        Mnorm[i,j] = norm(Mag)
        params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm[i,j])")

        m_dict["$(i),$(j)"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm[i,j])
    end

    M_mean = sum(Mnorm) / nphys
    params.verbosity >= 4 && println("|M|_mean = $(M_mean)")
    return M_mean, m_dict
end

function magnetization_value(model::Heisenberg{Kagome{:merge}}, A, env::VUMPSEnv, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack ifparallel = params.boundary_alg
    @unpack forloop_iter = params
    atype = _arraytype(ACu[1])
    etype = eltype(ACu[1])
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))
    d = size(Sx, 1)
    Id = atype(Matrix{Float64}(I, d, d))

    Sx1 = reshape((@tensor Sx1[1,2,3,4,5,6] := Sx[1,4] * Id[2,5] * Id[3,6]), d^3, d^3)
    Sy1 = reshape((@tensor Sy1[1,2,3,4,5,6] := Sy[1,4] * Id[2,5] * Id[3,6]), d^3, d^3)
    Sz1 = reshape((@tensor Sz1[1,2,3,4,5,6] := Sz[1,4] * Id[2,5] * Id[3,6]), d^3, d^3)

    Sx2 = reshape((@tensor Sx2[1,2,3,4,5,6] := Id[1,4] * Sx[2,5] * Id[3,6]), d^3, d^3)
    Sy2 = reshape((@tensor Sy2[1,2,3,4,5,6] := Id[1,4] * Sy[2,5] * Id[3,6]), d^3, d^3)
    Sz2 = reshape((@tensor Sz2[1,2,3,4,5,6] := Id[1,4] * Sz[2,5] * Id[3,6]), d^3, d^3)

    Sx3 = reshape((@tensor Sx3[1,2,3,4,5,6] := Id[1,4] * Id[2,5] * Sx[3,6]), d^3, d^3)
    Sy3 = reshape((@tensor Sy3[1,2,3,4,5,6] := Id[1,4] * Id[2,5] * Sy[3,6]), d^3, d^3)
    Sz3 = reshape((@tensor Sz3[1,2,3,4,5,6] := Id[1,4] * Id[2,5] * Sz[3,6]), d^3, d^3)

    Ni, Nj = size(ACu)
    len = length(ACu.data)
    m_dict = Dict{String, Any}()
    Mnorm1 = zeros(etype, Ni, Nj)
    Mnorm2 = zeros(etype, Ni, Nj)
    Mnorm3 = zeros(etype, Ni, Nj)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        Mx1 = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx1; ifparallel, forloop_iter)
        My1 = etype == Float64 ? 0.0 : contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy1; ifparallel, forloop_iter)
        Mz1 = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz1; ifparallel, forloop_iter)

        Mx2 = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx2; ifparallel, forloop_iter)
        My2 = etype == Float64 ? 0.0 : contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy2; ifparallel, forloop_iter)
        Mz2 = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz2; ifparallel, forloop_iter)

        Mx3 = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx3; ifparallel, forloop_iter)
        My3 = etype == Float64 ? 0.0 : contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy3; ifparallel, forloop_iter)
        Mz3 = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz3; ifparallel, forloop_iter)

        n = contract_n_11(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]; ifparallel, forloop_iter)

        Mag1 = [Mx1/n, My1/n, Mz1/n]
        Mnorm1[i,j] = norm(Mag1)
        params.verbosity >= 4 && println("M1 = $(Mag1)\n|M1| = $(Mnorm1)")
        Mag2 = [Mx2/n, My2/n, Mz2/n]
        Mnorm2[i,j] = norm(Mag2)
        params.verbosity >= 4 && println("M2 = $(Mag2)\n|M2| = $(Mnorm2)")
        Mag3 = [Mx3/n, My3/n, Mz3/n]
        Mnorm3[i,j] = norm(Mag3)
        params.verbosity >= 4 && println("M3 = $(Mag3)\n|M3| = $(Mnorm3)")

        m_dict["$(i),$(j),1"] = Dict("Mx" => Mag1[1], "My" => Mag1[2], "Mz" => Mag1[3], "|M|" => Mnorm1[i,j])
        m_dict["$(i),$(j),2"] = Dict("Mx" => Mag2[1], "My" => Mag2[2], "Mz" => Mag2[3], "|M|" => Mnorm2[i,j])
        m_dict["$(i),$(j),3"] = Dict("Mx" => Mag3[1], "My" => Mag3[2], "Mz" => Mag3[3], "|M|" => Mnorm3[i,j])
    end

    M_mean = sum(Mnorm1 + Mnorm2 + Mnorm3)/(3*len)
    params.verbosity >= 4 && println("|M|_mean = $(M_mean)")
    return M_mean, m_dict
end
