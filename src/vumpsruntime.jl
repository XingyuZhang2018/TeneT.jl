@kwdef mutable struct VUMPS
    ifupdown::Bool = true
    ifdownfromup::Bool = true
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    miniter::Int = Defaults.miniter
    show_every::Int = 10
    verbosity::Int = Defaults.verbosity
end

function VUMPSRuntime(M, χ::Int)
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    _, FL = leftenv(AL, conj.(AL), M)
    _, FR = rightenv(AR, conj.(AR), M)
    C = LRtoC(L, R)
    return VUMPSRuntime(AL, AR, C, FL, FR)
end

function _down_M(M)
    Ni, Nj = size(M)
    Md = Zygote.Buffer(M)
    for j in 1:Nj, i in 1:Ni
        ir = Ni + 1 - i
        Md[i, j] = permutedims(conj(M[ir, j]), (1,4,3,2))
    end
    return copy(Md)
end

function VUMPSRuntime(M, χ::Int, alg::VUMPS)
    Ni, Nj = size(M)

    rtup = VUMPSRuntime(M, χ)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"

    if alg.ifupdown     
        if alg.ifdownfromup
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) from up(↑) environment"
            return rtup
        else
            Md = _down_M(M)
            rtdown = VUMPSRuntime(Md, χ)
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) environment"
            return rtup, rtdown
        end
    else
        return rtup
    end
end

function vumps_itr(rt::VUMPSRuntime, M, alg::VUMPS)
    t = Zygote.@ignore time()
    for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end
    return rt
end

function leading_boundary(rt::VUMPSRuntime, M, alg::VUMPS)
    rtup = vumps_itr(rt, M, alg)
    if alg.ifdownfromup
        Md = _down_M(M)
        rtdown = vumps_itr(rtup, Md, alg)
        return rtup, rtdown
    else
        return rtup
    end
end

function VUMPSEnv(rt::VUMPSRuntime, ::Matrix)
    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    return VUMPSEnv(AC, AR, AC, AR, FL, FR, FL, FR)
end

function leading_boundary(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M, alg::VUMPS)
    rtup, rtdown = rt
    
    rtup = vumps_itr(rtup, M, alg)

    Md = _down_M(M)
    rtdown = vumps_itr(rtdown, Md, alg)
    return rtup, rtdown
end

function VUMPSEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M)
    rtup, rtdown = rt

    ALu, ARu, Cu, FLu, FRu = rtup.AL, rtup.AR, rtup.C, rtup.FL, rtup.FR
    ACu = ALCtoAC(ALu, Cu)

    ALd, ARd, Cd = rtdown.AL, rtdown.AR, rtdown.C
    ACd = ALCtoAC(ALd, Cd)

    _, FLo =  leftenv(ALu, conj.(ALd), M, FLu; ifobs = true)
    _, FRo = rightenv(ARu, conj.(ARd), M, FRu; ifobs = true)
    return VUMPSEnv(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
end

function vumps_step(rt::VUMPSRuntime, M, alg::VUMPS)
    verbosity = alg.verbosity
    @unpack AL, C, AR, FL, FR = rt
    AC = Zygote.@ignore ALCtoAC(AL,C)
    _, ACp = ACenv(AC, FL, M, FR; verbosity)
    _,  Cp =  Cenv( C, FL, FR; verbosity)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    _, FL =  leftenv(AL, conj.(ALp), M, FL; verbosity)
    _, FR = rightenv(AR, conj.(ARp), M, FR; verbosity)
    _, ACp = ACenv(ACp, FL, M, FR; verbosity)
    _,  Cp =  Cenv( Cp, FL, FR; verbosity)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    return VUMPSRuntime(ALp, ARp, Cp, FL, FR), err
end