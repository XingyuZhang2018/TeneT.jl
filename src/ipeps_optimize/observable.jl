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
function observable(A, χ, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps, _obs_callback=nothing)
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
    write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$(D)"), params)

    # Visualization
    if _obs_callback !== nothing
        _obs_callback(e, mag, ξ, χ)
    elseif params.ifplot
        # Standalone call: one-shot lattice plot (no convergence accumulation)
        obs_path = joinpath(params.folder, "D$(D)", "observable")
        plot_lattice_obs(e[2], mag[2], params.model.lattice, params.pattern;
                         save_path=obs_path, save_format=params.plot_format, χ=χ)
    end

    if params.model.lattice == Honeycomb(:brickwall)
        Wp_value(params.model, A, env, params)
    end
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
        jr = mod1(j + 1, Nj)
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
# f-wave pRVB (Kekulé VBS) order parameter for honeycomb brickwall iPEPS
# ============================================================================
# Measures dimerization on hexagonal plaquettes.
# Hexagon layout (brickwall coordinates, starting at (i,j)):
#
#   (i,j)---(i,j+1)---(i,j+2)
#     |                   |
#   (ir,j)--(ir,j+1)--(ir,j+2)
#
# 6 NN bonds clockwise:
#   B1: (i,j)-(i,j+1)       horizontal
#   B2: (i,j+1)-(i,j+2)     horizontal
#   B3: (i,j+2)-(ir,j+2)    vertical
#   B4: (ir,j+1)-(ir,j+2)   horizontal
#   B5: (ir,j)-(ir,j+1)     horizontal
#   B6: (i,j)-(ir,j)        vertical
#
# Kekulé K₁ = {B1, B3, B5},  K₂ = {B2, B4, B6}
# Δf = avg(K₁) - avg(K₂)
# ============================================================================
function fwave_order(model::HamiltonianModel, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    forloop_iter = params.boundary_alg.forloop_iter

    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))

    function bond_H(i, j)
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        return e / n
    end

    function bond_V(i, j)
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        return e / n
    end

    function hexagon_bonds(i, j)
        j1 = j
        j2 = mod1(j + 1, Nj)
        j3 = mod1(j + 2, Nj)
        ir = mod1(i + 1, Ni)
        B = zeros(ComplexF64, 6)
        B[1] = bond_H(i,  j1)   # (i,j)-(i,j+1)
        B[2] = bond_H(i,  j2)   # (i,j+1)-(i,j+2)
        B[3] = bond_V(i,  j3)   # (i,j+2)-(ir,j+2)
        B[4] = bond_H(ir, j2)   # (ir,j+1)-(ir,j+2)
        B[5] = bond_H(ir, j1)   # (ir,j)-(ir,j+1)
        B[6] = bond_V(i,  j1)   # (i,j)-(ir,j)
        return B
    end

    # Hexagon 1: starting at (1,2)
    bonds1 = hexagon_bonds(1, 2)
    Δf1 = (bonds1[1] + bonds1[3] + bonds1[5]) / 3 - (bonds1[2] + bonds1[4] + bonds1[6]) / 3

    # Hexagon 2: starting at (2,1)
    bonds2 = hexagon_bonds(2, 1)
    Δf2 = (bonds2[1] + bonds2[3] + bonds2[5]) / 3 - (bonds2[2] + bonds2[4] + bonds2[6]) / 3

    params.verbosity >= 3 && println("f-wave order parameter:")
    params.verbosity >= 3 && println("  Hexagon 1 bonds: ", real.(bonds1))
    params.verbosity >= 3 && println("  Hexagon 2 bonds: ", real.(bonds2))
    params.verbosity >= 3 && println("  Δf1 = $(real(Δf1)),  Δf2 = $(real(Δf2))")

    @show Δf1 Δf2
    return (Δf1, Δf2), (bonds1, bonds2)
end