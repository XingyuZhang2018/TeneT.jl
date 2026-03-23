# Observable computation for iPEPS
# Magnetization, correlation length, and full observable wrapper

using Printf

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
    rt = initialize_vumps_runtime(A, D, χ, params; restriction_ipeps)

    A = restriction_ipeps(A)
    A = build_A(A, params)
    M = build_M(A, params)

    rt, _ = leading_boundary(rt, M, params.boundary_alg)
    params.ifsave_env && save_rt(joinpath(params.folder, "D$(D)", "VUMPS_rt_env"), rt; file="χ$(χ).jld2")
    env = VUMPSEnv(rt, M, params.boundary_alg)
    e = expectation_value(params.model, A, env, params)
    mag = magnetization_value(params.model, A, env, params)
    ξ = cor_len_value(env, params)
    write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$(D)"), params)
    return e, mag, ξ
end

# ============================================================================
# Magnetization — generic (Heisenberg, J1J2, J1J2J3, SS)
# ============================================================================

function magnetization_value(model, A, env::VUMPSEnv, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Ni, Nj = size(ACu)
    len = length(ACu.data)
    forloop_iter = params.forloop_iter
    m_dict = Dict{String, Any}()
    Mnorm = zeros(ComplexF64, Ni, Nj)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        Mx = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx; forloop_iter)
        My = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy; forloop_iter)
        Mz = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz; forloop_iter)

        n = contract_n1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]; forloop_iter)
        Mag = [Mx/n, My/n, Mz/n]
        Mnorm[i,j] = norm(Mag)
        params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")

        m_dict["$(i),$(j)"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm[i,j])
    end

    M_mean = sum(Mnorm)/len
    params.verbosity >= 4 && println("|M|_mean = $(M_mean)")
    return M_mean, m_dict
end

function magnetization_value(model, A, env::CTMEnv, params)
    @unpack C, T = env
    @unpack forloop_iter = params
    m_dict = Dict{String, Any}()
    etype = eltype(A)
    atype = _arraytype(A)

    To = CTCtoT(C, T)
    Sx = atype(const_Sx(model.S))
    Sy = atype(const_Sy(model.S))
    Sz = atype(const_Sz(model.S))

    Mx = contract_o1(To, T, A, Sx; forloop_iter)
    My = etype == Float64 ? 0.0 : contract_o1(To, T, A, Sy; forloop_iter)
    Mz = contract_o1(To, T, A, Sz; forloop_iter)

    n = contract_n1(To, T, A; forloop_iter)
    Mag = [Mx/n, My/n, Mz/n]
    Mnorm = norm(Mag)
    params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")
    m_dict["1,1"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm)

    return Mnorm, m_dict
end

# ============================================================================
# Magnetization — Kagome (3 sublattice sites per unit cell)
# ============================================================================

function magnetization_value(model::Kagome, A, env::VUMPSEnv, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
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
    forloop_iter = params.forloop_iter
    m_dict = Dict{String, Any}()
    Mnorm1 = zeros(ComplexF64, Ni, Nj)
    Mnorm2 = zeros(ComplexF64, Ni, Nj)
    Mnorm3 = zeros(ComplexF64, Ni, Nj)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        Mx1 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx1; forloop_iter)
        My1 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy1; forloop_iter)
        Mz1 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz1; forloop_iter)

        Mx2 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx2; forloop_iter)
        My2 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy2; forloop_iter)
        Mz2 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz2; forloop_iter)

        Mx3 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sx3; forloop_iter)
        My3 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sy3; forloop_iter)
        Mz3 = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j], Sz3; forloop_iter)

        n = contract_n1(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j]; forloop_iter)

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

function Cmap(C, Aui, Adi, J::Int)
    Nj = size(Aui, 1)
    for j = 1:Nj
        jr = mod1(J+j-1, Nj)
        C = ρmap(C, Aui[jr], Adi[jr])
    end
    return C
end

"""
    cor_len_value(env::VUMPSEnv, params)

Compute the correlation length from the transfer matrix eigenvalues
of the VUMPS environment.
"""
function cor_len_value(env::VUMPSEnv, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Cint = cellones(ACu)[1]
    λcs, _, info = eigsolve(C->Cmap(C, ARu[1,:], conj(ARu[1,:]), 1), Cint, 10, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    @show λcs λcs[2]/λcs[1]
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    @show ξ
    params.verbosity >= 4 && println("ξ = $(ξ)")
    return ξ
end

"""
    cor_len_value(env::CTMEnv, params)

Compute the correlation length from the CTM corner transfer matrix.
"""
function cor_len_value(env::CTMEnv, params)
    @unpack C, T = env

    λcs, _, info = eigsolve(C->Lmap(C, T, conj(T)), C, 10, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    @show λcs λcs[2]/λcs[1]
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    @show ξ
    params.verbosity >= 4 && println("ξ = $(ξ)")
    return ξ
end

# ============================================================================
# Observable log writer
# ============================================================================

"""
    write_obs_log(e, mag, ξ, χ, folder, ::iPEPSOptimize)

Write energy, magnetization, and correlation length to a log file.
"""
function write_obs_log(e, mag, ξ, χ, folder, ::iPEPSOptimize)
    path = joinpath(folder, "observable")
    isdir(path) || mkpath(path)
    obs_log = joinpath(path, "χ$χ.log")
    e_dict = e[2]
    m_dict = mag[2]

    open(obs_log, "w") do io
        write(io, @sprintf("energy_per_site:\n%.15f\n", real(e[1])))

        for bond_type in keys(e_dict)
            write(io, "$bond_type: i j energy\n")
            for pos in keys(e_dict[bond_type])
                write(io, @sprintf("%s %.15f\t", pos, real(e_dict[bond_type][pos])))
            end
            write(io, "\n")
        end

        write(io, @sprintf("magnetization_norm_per_site:\n%.15f\n", real(mag[1])))

        write(io, "magnetization: i j |M| Mx My Mz\n")
        for pos in keys(m_dict)
            write(io, @sprintf("%s %.15f %.15f %.15f %.15f\t", pos, real(m_dict[pos]["|M|"]), real(m_dict[pos]["Mx"]), real(m_dict[pos]["My"]), real(m_dict[pos]["Mz"])))
            write(io, "\n")
        end
        write(io, @sprintf("correlation_length:\n%.15f\n", real(ξ)))
    end
end
