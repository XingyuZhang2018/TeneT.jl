@kwdef mutable struct CTMRG{LT}
    tol::Float64 = 1e-10                # CTMRG error tolerance
    maxiter::Int = 10                   # maxiter iteration of CTMRG without AD
    miniter::Int = 1                    # miniter iteration of CTMRG without AD
    maxiter_ad::Int = 10                # maxiter iteration of CTMRG with AD
    miniter_ad::Int = 3                 # miniter iteration of CTMRG with AD
    forloop_iter::Int = 1               # the iteration of the for-loop contraction, when > 1, automatically use checkpoint
    show_every::Int = 10                # show the iteration result at every n iterations
    verbosity::Int = Defaults.verbosity # verbosity control the output message

    ifcheckpoint::Bool = false          # if checkpoint at every iteration
end

function init_Runtime(M, χ::Int, ::CTMRG{:honeycomb})
    Ta = [randSA(M[1], [(D = size(m, 1); (χ, D, D, χ)) for m in M[1].data]) for _ in 1:3]
    Tb = [randSA(M[1], [(D = size(m, 1); (χ, D, D, χ)) for m in M[1].data]) for _ in 1:3]
    C = [randSA(M[1], [(χ, χ) for m in M.data]) for _ in 1:3]
    return CTMEnv{:honeycomb}(Ta, Tb, C)
end

function ctmrg_itr(rt::CTMRGRuntime, M, alg::CTMRG)
    t = Zygote.@ignore time()

    atype = _arraytype(M)
    id = get_device_id(atype)
    local err
    Zygote.@ignore alg.verbosity >= 2 && @info "Start CTMRG iteration at $(get_device(atype)) without AD..."
    Zygote.@ignore for i in 1:alg.maxiter
        rt, err = ctmrg_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("CTMRG@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("CTMRG conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("CTMRG cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        end
    end

    Zygote.@ignore alg.verbosity >= 2 && @info "Start CTMRG iteration at $(get_device(atype)) with AD..."
    for i in 1:alg.maxiter_ad
        rt, err = alg.ifcheckpoint ? checkpoint(ctmrg_step, rt, M, alg) : ctmrg_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("CTMRG@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("CTMRG conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("CTMRG cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        end
    end

    return rt, err
end

function leading_boundary(rt::CTMRGRuntime, M, alg::CTMRG)
    rt, err = ctmrg_itr(rt, M, alg)
    return rt, err
end

function ctmrg_step(rt::CTMRGRuntime, M, alg::CTMRG{:honeycomb})
    @unpack Ta, Tb, C = rt

    Ua, Ub = ctmrg_get_isometries(Ta, Tb, C, M, alg)
    Ta, Tb = T_update(Ta, Tb, Ua, Ub, M, alg)
    C = C_update(C, Ta, Tb, alg)

    return CTMRGRuntime(Ta, Tb, C), err
end