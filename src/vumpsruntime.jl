@kwdef mutable struct VUMPS
    tol::Float64 = 1e-10                # VUMPS error tolerance
    maxiter::Int = 10                   # maxiter iteration of VUMPS without AD
    miniter::Int = 1                    # miniter iteration of VUMPS without AD
    maxiter_ad::Int = 10                # maxiter iteration of VUMPS with AD
    miniter_ad::Int = 3                 # miniter iteration of VUMPS with AD
    ifparallel::Bool = false            # if use parallel for-loop contraction
    forloop_iter::Int = 1               # the iteration of the for-loop contraction, when > 1, automatically use checkpoint
    power_iter::Int = 5                 # the iteration of the power method, only works when `ifsimple_eig = true`
    power_iter_obs::Int = 20            # the iteration of the power method for the up and down observation environment, only works when `ifsimple_eig = true`
    show_every::Int = 10                # show the iteration result at every n iterations
    verbosity::Int = Defaults.verbosity # verbosity control the output message

    ifsimple_eig::Bool = true           # if use the simple power method as eigsolve
    ifcheckpoint::Bool = false          # if checkpoint at every iteration
    ifgpu_cpu_combo::Bool = false       # if save the environment on the CPU memory but calculate on the GPU # currently not implement
end

function init_VUMPSRuntime(M, χ::Int, alg::VUMPS)
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    _, FL = leftenv(AL, conj(AL), M; alg)
    C = LRtoC(L, L)
    return VUMPSRuntime(AL, C, FL)
end

_down_m(m::leg4) = permutedims(m, (1,4,3,2))
_down_m(m::leg5) = permutedims(m, (1,4,3,2,5))
_down_m(m::leg8) = permutedims(m, (1,2,7,8,5,6,3,4))
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
    @unpack AL, C, FL = rtup
    # Ni = size(AL, 1)
    # index = [Ni + 1 - i for i in 1:Ni]
    ALd = StructArray(AL.data, Md.pattern)
    Cd = StructArray(C.data, Md.pattern)
    FLd = StructArray(FL.data, Md.pattern)
    return VUMPSRuntime(ALd, Cd, FLd)
end

function VUMPSRuntime(M::StructArray, χ::Int, alg::VUMPS)
    Ni, Nj = size(M)

    rtup = init_VUMPSRuntime(M, χ, alg)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"

    return rtup
end

function vumps_itr(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    t = Zygote.@ignore time()

    atype = _arraytype(M)
    id = get_device_id(atype)
    local err
    Zygote.@ignore alg.verbosity >= 2 && @info "Start VUMPS iteration at $(get_device(atype)) without AD..."
    Zygote.@ignore for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        end
    end

    Zygote.@ignore alg.verbosity >= 2 && @info "Start VUMPS iteration at $(get_device(atype)) with AD..."
    for i in 1:alg.maxiter_ad
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step, rt, M, alg) : vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        end
    end

    return rt, err
end

function leading_boundary(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    rt, err = vumps_itr(rt, M, alg)
    return rt, err
end

function VUMPSEnv(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    @unpack AL, C, FL = rt
    _, FLo =  leftenv(AL, AL, M, FL; ifobs = true, alg)
    # return VUMPSEnv(AC, AR, ACd, ARd, FL, FR, FLo, FRo)
    return VUMPSEnv(AL, C, FL, FLo)
end

function vumps_step(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    @unpack AL, C, FL = rt
    AC = ALCtoAC(AL,C)
    _, FL =  leftenv(AL, conj(AL), M, FL; alg)
    _, AC = ACenv(AC, FL, M; alg)
    _,  C =  Cenv( C, FL; alg)
    AL, err = ACCtoAL(AC, C)
    C = for_gc(C)
    return VUMPSRuntime(AL, C, FL), err
end