@kwdef mutable struct VUMPS
    ifupdown::Bool = true
    ifdownfromup::Bool = false
    ifsimple_eig::Bool = Defaults.ifsimple_eig
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    miniter::Int = Defaults.miniter
    maxiter_ad::Int = Defaults.maxiter_ad
    miniter_ad::Int = Defaults.miniter_ad
    ifcheckpoint::Bool = Defaults.ifcheckpoint
    show_every::Int = 10
    verbosity::Int = Defaults.verbosity
end

function init_VUMPSRuntime(M, χ, alg::VUMPS)
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    _, FL =  leftenv(AL, adjoint(AL), M; alg)
    _, FR = rightenv(AR, adjoint(AR), M; alg)
    C = LRtoC(L, R)
    return VUMPSRuntime(AL, AR, C, FL, FR)
end

_down_m(m::ipeps) = _fit_spaces(permute(m', ((2,5,4,3), (1,))), m)
_down_m(m::bulk) = _fit_spaces(permute(m', ((3,2), (1,4))), m)
function _down_M(M::StructArray)
    Ni, Nj = size(M)
    pattern_d = copy(M.pattern)
    Zygote.@ignore begin
        @inbounds for i in 1:Ni, j in 1:Nj
            ir = Ni + 1 - i
            pattern_d[i, j] = M.pattern[ir, j]
        end
    end
    data_d = [_down_m(data) for data in M.data]
    Md = StructArray(data_d, pattern_d)
    return Md
end

function _down_init_from_up(rtup::VUMPSRuntime, Md::StructArray)
    @unpack AL, AR, C, FL, FR = rtup
    Ni = size(AL, 1)
    # index = [Ni + 1 - i for i in 1:Ni]
    ALd = StructArray(AL.data, Md.pattern)
    ARd = StructArray(AR.data, Md.pattern)
    Cd = StructArray(C.data, Md.pattern)
    FLd = StructArray(FL.data, Md.pattern)
    FRd = StructArray(FR.data, Md.pattern)
    return VUMPSRuntime(ALd, ARd, Cd, FLd, FRd)
end

function VUMPSRuntime(M::StructArray, χ, alg::VUMPS)
    Ni, Nj = size(M)

    rtup = init_VUMPSRuntime(M, χ, alg)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"

    if alg.ifupdown    
        Md = _down_M(M) 
        if alg.ifdownfromup
            rtdown = _down_init_from_up(rtup, Md)
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) from up(↑) environment"
            return rtup, rtdown
        else
            rtdown = init_VUMPSRuntime(Md, χ, alg)
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) environment"
            return rtup, rtdown
        end
    else
        return rtup
    end
end

function vumps_itr(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    t = Zygote.@ignore time()

    Zygote.@ignore alg.verbosity >= 2 && @info @sprintf("Start VUMPS iteration without AD...")
    Zygote.@ignore for i in 1:alg.maxiter
        rt, err = vumps_step_power(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end

    Zygote.@ignore alg.verbosity >= 2 && @info @sprintf("Start VUMPS iteration with AD...")
    for i in 1:alg.maxiter_ad
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step_power, rt, M, alg) : vumps_step_power(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end

    return rt
end

function leading_boundary(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    rt = vumps_itr(rt, M, alg)
    return rt
end

function VUMPSEnv(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    return VUMPSEnv(AC, AR, AC, AR, FL, FR, FL, FR)
end

function leading_boundary(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M::StructArray, alg::VUMPS)
    rtup, rtdown = rt
    
    rtup = vumps_itr(rtup, M, alg)

    Md = _down_M(M)
    rtdown = vumps_itr(rtdown, Md, alg)
    return rtup, rtdown
end

function VUMPSEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M::StructArray, alg)
    rtup, rtdown = rt

    ALu, ARu, Cu, FLu, FRu = rtup.AL, rtup.AR, rtup.C, rtup.FL, rtup.FR
    ACu = ALCtoAC(ALu, Cu)

    ALd, ARd, Cd = rtdown.AL, rtdown.AR, rtdown.C
    ACd = ALCtoAC(ALd, Cd)

    _, FLo =  leftenv(ALu, adjoint(ALd), M, FLu; ifobs = true, alg)
    _, FRo = rightenv(ARu, adjoint(ARd), M, FRu; ifobs = true, alg)
    return VUMPSEnv(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
end

function vumps_step_power(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    @unpack AL, C, AR, FL, FR = rt
    AC = ALCtoAC(AL,C)
    _, ACp = ACenv(AC, FL, FR, M; alg)
    _,  Cp =  Cenv( C, FL, FR; alg)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    _, FL =  leftenv(AL, adjoint(ALp), M, FL; alg)
    _, FR = rightenv(AR, adjoint(ARp), M, FR; alg)
    _, ACp = ACenv(ACp, FL, FR, M; alg)
    _,  Cp =  Cenv( Cp, FL, FR; alg)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    return VUMPSRuntime(ALp, ARp, Cp, FL, FR), err
end

function vumps_step_Hermitian(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    @unpack AL, C, AR, FL, FR = rt
    AC = ALCtoAC(AL,C)
    _, FL =  leftenv(AL, adjoint(AL), M, FL; alg)
    _, FR = rightenv(AR, adjoint(AR), M, FR; alg)
    _, AC = ACenv(AC, FL, FR, M; alg)
    _,  C =  Cenv( C, FL, FR; alg)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end