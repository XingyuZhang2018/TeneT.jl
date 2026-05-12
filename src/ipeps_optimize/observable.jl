# Observable computation for iPEPS
# Magnetization, correlation length, and full observable wrapper

"""
    energy(A, env, params::iPEPSOptimize)

Main entry point for energy computation during iPEPS optimization.
Calls `energy_value` dispatching on `params.model` type.
"""
function energy(A, env, params::iPEPSOptimize)
    return energy_value(params.model, A, env, params)[1]
end

# ============================================================================
# Full observable computation
# ============================================================================

"""
    observable(A, χ, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps)

Compute all observables (energy, magnetization, correlation length) for a
given iPEPS tensor `A` at bond dimension `χ`. Initializes a VUMPS runtime,
converges the boundary, and evaluates expectation values.
"""
function observable(A, χ, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps)
    D = size(A, 1)
    rt = initialize_env(A, D, χ, params; restriction_ipeps)

    _G_cache[] = nothing
    A = restriction_ipeps(A)
    A = build_A(A, params)

    rt, _ = leading_boundary(rt, A, params.boundary_alg)
    params.ifsave_env && save_rt(joinpath(params.folder, "D$(D)", "environment"), rt; file="χ$(χ).jld2")
    env = ObsEnv(rt, A, params.boundary_alg)
    e = energy_value(params.model, A, env, params)
    mag = magnetization_value(params.model, A, env, params)
    ξ = cor_len_value(env, params)

    # For Kagome merge: compute per-bond energies and use them for logging/plotting
    if params.model.lattice isa Kagome{:merge}
        e_perbond = energy_value_perbond(params.model, A, env, params)
        e = (e[1], e_perbond)  # replace aggregate e_dict with per-bond e_dict
    end

    write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$(D)"), params)

    # Visualization: read all logs and plot (includes history from previous runs)
    if params.ifplot
        obs_path = joinpath(params.folder, "D$(D)", "observable")
        plot_observables(obs_path, params.model.lattice, params.pattern;
                         save_format=params.plot_format, S=params.model.S)
    end

    # if params.model.lattice == Honeycomb(:brickwall_h)
    #     Wp_value(params.model, A, env, params)
    #     fwave_order(params.model, A, env, params)
    # end
    return e, mag, ξ
end

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
    ir_oneside(i) = _oneside_down_index(typeof(model), i, Ni)
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

# ============================================================================
# Correlation length
# ============================================================================

"""
    cor_len_value(env::VUMPSEnv, params)

Compute the correlation length from the transfer matrix eigenvalues
of the VUMPS environment.
"""
function cor_len_value(env::VUMPSEnv, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Cint = cellones(ACu)[1]
    λcs, _, info = eigsolve(C->Lmap(1, C, ARu[1,:], ARd[1,:]), Cint, 10, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ = $(ξ)")
    return ξ
end

function cor_len_value(env::PlaquetteVUMPSEnv, params)
    @unpack AL, C, FLu, FLo = env
    Cint = cellones(AL)[1]
    λcs, _, info = eigsolve(C -> Rmap(1, C, AL[1,:], conj(AL[1,:])), Cint, 10, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end
    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ = $(ξ)")
    return ξ
end

function cor_len_value(env::OnesideVUMPSEnv, params)
    @unpack AC, AR, FLu, FRu, FLo, FRo = env
    Cint = cellones(AC)[1]
    model = params.model
    Ni = size(AC, 1)
    ir = _oneside_down_index(typeof(model), 1, Ni)
    λcs, _, info = eigsolve(C -> Lmap(1, C, AR[1,:], AR[ir,:]), Cint, 10, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ = $(ξ)")
    return ξ
end

function cor_len_value(env::C4vVUMPSEnv, params)
    @unpack AL, C = env

    λcs, _, info = eigsolve(C->Lmap(C, AL, conj(AL)), C, 10, :LM; maxiter=100, ishermitian = false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end
        
    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ = $(ξ)")

    return ξ
end

"""
    cor_len_value(env::CTMEnv, params)

Compute the correlation length from the CTM corner transfer matrix.
"""
function cor_len_value(env::CTMEnv, params)
    @unpack C, T = env

    λcs, _, info = eigsolve(C->Lmap(C, T, T), C, 10, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ = $(ξ)")
    return ξ
end

# ============================================================================
# Wp value for iPEPS with brickwall unit cell 
# ============================================================================
function Wp_value(model::HamiltonianModel, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack ifparallel = params.boundary_alg
    @unpack forloop_iter = params.boundary_alg
    atype = _arraytype(ACu[1])
    S = model.S
    iSx = Zygote.@ignore atype(exp(1im * pi * const_Sx(S)))
    iSy = Zygote.@ignore atype(exp(1im * pi * const_Sy(S)))
    iSz = Zygote.@ignore atype(exp(1im * pi * const_Sz(S)))

    Ni, Nj = size(ACu)
    Ni,Nj = size(ACu)
    i, j = 1, 2
    ir = mod1(i + 1, Ni)
    id = mod1(Ni - i, Ni) 
    jr = mod1(j + 1, Nj)
    jrr = mod1(j + 2, Nj)

    o = contract_o_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], iSz, iSy, iSx, iSx, iSy, iSz; ifparallel, forloop_iter)
    n = contract_n_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr]; ifparallel, forloop_iter)
    Wp1 = o/n

    i, j = 2, 1
    ir = mod1(i + 1, Ni)
    id = mod1(Ni - i, Ni) 
    jr = mod1(j + 1, Nj)
    jrr = mod1(j + 2, Nj)

    o = contract_o_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], iSz, iSy, iSx, iSx, iSy, iSz; ifparallel, forloop_iter)
    n = contract_n_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr]; ifparallel, forloop_iter)
    Wp2 = o/n
    @show Wp1 Wp2
    return Wp1, Wp2
end

# ============================================================================
# f-wave pRVB order parameter via ring exchange ⟨K₆⟩ = ⟨C₆ + C₆⁻¹⟩
# ============================================================================
# K₆ is the ring exchange operator on a hexagonal plaquette.
# The f-wave pRVB state is an eigenstate of K₆.
# ⟨K₆⟩ ≠ 0 indicates ring-exchange coherence (f-wave character).
#
# Grid layout (2×3 plaquette, sites 1-6):
#
#   site1---site2---site3     (top row: A[i,j], A[i,jr], A[i,jrr])
#     |                 |
#   site4---site5---site6     (bottom row: A[ir,j], A[ir,jr], A[ir,jrr])
#
# Hexagonal ring (clockwise): 1 → 2 → 3 → 6 → 5 → 4 → 1
#
# C₆ shifts spins one step along the ring:
#   grid source = [2, 3, 6, 1, 4, 5]
#   i.e. site 1 ← site 2, site 2 ← site 3, site 3 ← site 6, etc.
#
# Decomposition: C₆ = Σ_{s} ⊗ₖ |s[source[k]]⟩⟨s[k]|  (64 terms)
# ⟨K₆⟩ = 2 Re(⟨C₆⟩)
# ============================================================================
function fwave_order(model::HamiltonianModel, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack ifparallel = params.boundary_alg
    @unpack forloop_iter = params.boundary_alg
    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)

    # Projector basis: proj[a,b] = |a⟩⟨b|  (a,b ∈ {1,2}, 1=↑, 2=↓)
    proj = Zygote.@ignore [atype(Float64[(i == a) * (j == b) for i in 1:2, j in 1:2])
                           for a in 1:2, b in 1:2]

    # C₆ source mapping in grid indices (hexagonal ring clockwise)
    c6_src = [2, 3, 6, 1, 4, 5]

    function compute_K6(i, j)
        ir  = mod1(i + 1, Ni)
        id  = mod1(Ni - i, Ni)
        jr  = mod1(j + 1, Nj)
        jrr = mod1(j + 2, Nj)

        n = contract_n_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j],
                          FRu[i,jrr], FRo[ir,jrr],
                          ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr],
                          A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr];
                          ifparallel, forloop_iter)

        # Sum over 2⁶ = 64 spin configurations for ⟨C₆⟩
        C6_val = ComplexF64(0)
        for idx in 0:63
            s1 = (idx       & 1) + 1
            s2 = ((idx >> 1) & 1) + 1
            s3 = ((idx >> 2) & 1) + 1
            s4 = ((idx >> 3) & 1) + 1
            s5 = ((idx >> 4) & 1) + 1
            s6 = ((idx >> 5) & 1) + 1
            s = (s1, s2, s3, s4, s5, s6)

            # Operator at grid site k: |s[c6_src[k]]⟩⟨s[k]|
            o = contract_o_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j],
                              FRu[i,jrr], FRo[ir,jrr],
                              ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr],
                              A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr],
                              proj[s[c6_src[1]], s[1]],
                              proj[s[c6_src[2]], s[2]],
                              proj[s[c6_src[3]], s[3]],
                              proj[s[c6_src[4]], s[4]],
                              proj[s[c6_src[5]], s[5]],
                              proj[s[c6_src[6]], s[6]];
                              ifparallel, forloop_iter)
            C6_val += o
        end

        return 2 * real(C6_val / n)
    end

    # Two inequivalent hexagonal plaquettes
    K6_1 = compute_K6(1, mod1(4, Nj))
    K6_2 = compute_K6(2, 1)

    params.verbosity >= 3 && println("f-wave ring exchange ⟨K₆⟩:")
    params.verbosity >= 3 && println("  Hexagon (1,4): K₆ = $(K6_1)")
    params.verbosity >= 3 && println("  Hexagon (2,1): K₆ = $(K6_2)")

    @show K6_1 K6_2
    return K6_1, K6_2
end