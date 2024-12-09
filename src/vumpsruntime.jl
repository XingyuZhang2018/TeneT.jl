@kwdef mutable struct VUMPS{L<:AbstractLattice}
    ifupdown::Bool = true
    ifdownfromup::Bool = false
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    miniter::Int = Defaults.miniter
    show_every::Int = 10
    verbosity::Int = Defaults.verbosity
end

function init_VUMPSRuntime(M, χ::Int, ::VUMPS{Lattice}) where Lattice
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    AL′ = Zygote.Buffer(AL)
    Ni, Nj = size(M)
    for j in  1:Nj, i in 1:Ni
        AL′[i,j] = ein"abc->cba"(AR[i, mod1(j+1, Nj)]) 
    end
    AL′ = copy(AL′)
    _, FL = leftenv(AL, conj.(AL′), M)
    # _, FR = rightenv(AR, conj.(AR), M)
    C = LRtoC(L, R)
    return VUMPSRuntime{Lattice}(AL, AR, C, FL)
end

_down_m(m::leg4) = permutedims(conj(m), (1,4,3,2))
_down_m(m::leg5) = permutedims(conj(m), (1,4,3,2,5))
function _down_M(M)
    Ni, Nj = size(M)
    Md = Zygote.Buffer(M)
    for j in 1:Nj, i in 1:Ni
        ir = Ni + 1 - i
        Md[i, j] = _down_m(M[ir, j])
    end
    return copy(Md)
end

function VUMPSRuntime(M, χ::Int, alg::VUMPS)
    Ni, Nj = size(M)

    rtup = init_VUMPSRuntime(M, χ, alg)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"

    if alg.ifupdown     
        if alg.ifdownfromup
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) from up(↑) environment"
            return rtup
        else
            Md = _down_M(M)
            rtdown = init_VUMPSRuntime(Md, χ, alg)
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
    return rt
end

function leading_boundary(rt::VUMPSRuntime, M, alg::VUMPS)
    rtup = vumps_itr(rt, M, alg)
    if alg.ifupdown && alg.ifdownfromup
        Md = _down_M(M)
        rtdown = vumps_itr(rtup, Md, alg)
        return rtup, rtdown
    else
        return rtup
    end
end

function VUMPSEnv(rt::VUMPSRuntime{L}, M::Matrix) where L
    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    # _, FLo =  leftenv(AL, conj.(AL), M, FL; ifobs = true)
    # _, FRo = rightenv(AR, conj.(AR), M, FR; ifobs = true)
    return VUMPSEnv{L}(AC, AR, AC, AR, FL, FR, FL, FR)
end

function leading_boundary(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M, alg::VUMPS)
    rtup, rtdown = rt
    
    rtup = vumps_itr(rtup, M, alg)

    Md = _down_M(M)
    rtdown = vumps_itr(rtdown, Md, alg)
    return rtup, rtdown
end

function VUMPSEnv(rt::Tuple{VUMPSRuntime{L}, VUMPSRuntime{L}}, M) where L
    rtup, rtdown = rt

    ALu, ARu, Cu, FLu, FRu = rtup.AL, rtup.AR, rtup.C, rtup.FL, rtup.FR
    ACu = ALCtoAC(ALu, Cu)

    ALd, ARd, Cd = rtdown.AL, rtdown.AR, rtdown.C
    ACd = ALCtoAC(ALd, Cd)

    _, FLo =  leftenv(ALu, conj.(ALd), M, FLu; ifobs = true)
    _, FRo = rightenv(ARu, conj.(ARd), M, FRu; ifobs = true)
    return VUMPSEnv{L}(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
end

function vumps_step_power(rt::VUMPSRuntime{L}, M, alg::VUMPS{L}) where L
    verbosity = alg.verbosity
    @unpack AL, C, AR, FL = rt
    AC = ALCtoAC(AL,C)
    _, ACp = ACenv(AC, FL, M, FL; verbosity)
    _,  Cp =  Cenv( C, FL, FL; verbosity)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    ALp′ = Zygote.Buffer(ALp)
    Ni, Nj = size(M)
    for j in  1:Nj, i in 1:Ni
        ALp′[i,j] = ein"abc->cba"(ARp[i, mod1(j+1, Nj)]) 
    end
    ALp′ = copy(ALp′)
    _, FL =  leftenv(AL, conj.(ALp′), M, FL; verbosity)
    # _, FR = rightenv(AR, conj.(ARp), M, FR; verbosity)
    _, ACp = ACenv(ACp, FL, M, FL; verbosity)
    _,  Cp =  Cenv( Cp, FL, FL; verbosity)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    return VUMPSRuntime{L}(ALp, ARp, Cp, FL), err
end

function vumps_step_Hermitian(rt::VUMPSRuntime{L}, M, alg::VUMPS{L}) where L 
    verbosity = alg.verbosity
    @unpack AL, C, AR, FL = rt
    AC = ALCtoAC(AL,C)
    _, FL =  leftenv(AL, conj.(AL), M, FL; verbosity)
    # _, FR = rightenv(AR, conj.(AR), M, FR; verbosity)
    _, AC = ACenv(AC, FL, M, FL; verbosity)
    _,  C =  Cenv( C, FL, FL; verbosity)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    return VUMPSRuntime{L}(AL, AR, C, FL), err
end

function fix_gauge_vumps_step(rt::VUMPSRuntime{L}, M, alg::VUMPS{L}) where L
    rt′, err = vumps_step_Hermitian(rt, M, alg)
    ALu, ARu, Cu, FLu = rt.AL, rt.AR, rt.C, rt.FL
    ALd, ARd, Cd, FLd = rt′.AL, rt′.AR, rt′.C, rt′.FL

    # _, σ = rightCenv(ARu, conj.(ARd); ifobs=false, verbosity=alg.verbosity) 
    # U, _ = Zygote.@ignore qrpos(σ[1])
    # AL_gauged = [ein"(ba,bcd),ed -> ace"(U, ALd, U') for ALd in ALd]
    # AR_gauged = [ein"(ba,bcd),ed -> ace"(U, ARd, U') for ARd in ARd]
    #  C_gauged = [ein"(ba,bc),dc -> ad"(U, Cd, U') for Cd in Cd]
    # FL_gauged = [ein"(ba,bcd),ed -> ace"(U', FLd, U) for FLd in FLd]
    # FR_gauged = [ein"(ab,bcd),de -> ace"(U, FRd, U') for FRd in FRd]

    AL_gauged = ALd
    AR_gauged = ARd
    C_gauged = Cd   
    FL_gauged = FLd
    # FR_gauged = FRd
    λ1 = Zygote.@ignore [ALu ./ AL_gauged  for (AL_gauged, ALu) in zip(AL_gauged, ALu)]
    λ2 = Zygote.@ignore [ARu ./ AR_gauged  for (AR_gauged, ARu) in zip(AR_gauged, ARu)]
    λ3 = Zygote.@ignore [Cu ./ C_gauged for (C_gauged, Cu) in zip(C_gauged, Cu)] 
    λ4 = Zygote.@ignore [FLu ./ FL_gauged  for (FL_gauged, FLu) in zip(FL_gauged, FLu)]
    # λ5 = Zygote.@ignore [FRu ./ FR_gauged  for (FR_gauged, FRu) in zip(FR_gauged, FRu)]

    AL_gauged = [AL_gauged .* λ1 for (AL_gauged,λ1) in zip(AL_gauged,λ1)]
    AR_gauged = [AR_gauged .* λ2 for (AR_gauged,λ2) in zip(AR_gauged,λ2)]
    C_gauged = [C_gauged .* λ3 for (C_gauged,λ3) in zip(C_gauged,λ3)]
    FL_gauged = [FL_gauged .* λ4 for (FL_gauged,λ4) in zip(FL_gauged,λ4)]
    # FR_gauged = [FR_gauged .* λ5 for (FR_gauged,λ5) in zip(FR_gauged,λ5)]
    return VUMPSRuntime{L}(AL_gauged, AR_gauged, C_gauged, FL_gauged), err
end