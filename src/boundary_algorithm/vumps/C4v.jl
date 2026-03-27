function leftenv_c4v(ALu, ALd, M, FL; alg, kwargs...)
    power_iter = alg.power_iter
    ifparallel = alg.ifparallel
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    f(FL) = FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter)
    if alg.ifsimple_eig
        if ifcheckpoint
            λFLs, FLs = checkpoint(simple_eig, f, FL; power_iter)
        else
            λFLs, FLs = simple_eig(f, FL; power_iter)
        end
    else
        λFLs, FLs, info = eigsolve(f, FL, 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100,shermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "FLenv_c4v not converged"
    end

    return λFLs[1], FLs[1]
end

function ACenv_c4v(AC, FL, M; alg, kwargs...)
    power_iter = alg.power_iter
    ifparallel = alg.ifparallel
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    f(AC) = ACmap_parallel(AC, FL, FL, M; ifparallel, forloop_iter)
    if alg.ifsimple_eig
        if ifcheckpoint
            λACs, ACs = checkpoint(simple_eig, f, AC; power_iter)
        else
            λACs, ACs = simple_eig(f, AC; power_iter)
        end
    else
        λACs, ACs, info = eigsolve(f, AC, 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100,shermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv_c4v not converged"
    end

    return λACs[1], ACs[1]
end

function Cenv_c4v(C, FL; alg, kwargs...)
    power_iter = alg.power_iter
    ifcheckpoint = alg.ifcheckpoint

    if alg.ifsimple_eig
        λCs, Cs = simple_eig(C1j -> Cmap(1, C1j, FL[:,jr], FL[:,jr]), power_iter)
    else
        λCs, Cs, info = eigsolve(C1j -> Cmap(1, C1j, FL[:,jr], FL[:,jr]),C[1,j], 1, :LM;
            alg_rrule=GMRES(verbosity=-1), maxiter=100,ishermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "Cenv_plaq not converged"
    end
        
    return λCs[1], Cs[1]
end


# ── initialization ─────────────────────────────────────────

function init_C4vVUMPSEnv(M, χ::Int, alg::VUMPS{:C4v})
    D = size(M, 1)  
    if M isa leg4
        FL = rand!(similar(M,χ,D,χ))
        FL += conj(permutedims(FL, (3,2,1)))
    else
        FL = rand!(similar(M,χ,D,D,χ))
        FL += conj(permutedims(FL, (4,2,3,1)))
    end
    AL, C = qr(_to_front(FL))
    AL = reshape(_arraytype(FL)(AL), size(FL))

    return C4vVUMPSEnv(AL, C, FL)
end

"""
    vumps_step(rt::C4vVUMPSEnv, M, alg::VUMPS{:C4v})

One step of the plaquette VUMPS: leftenv → ACenv → Cenv → ACCtoAL.
Only uses left environments (no right canonical / right environment).
"""
function vumps_step(rt::C4vVUMPSEnv, M::StructArray, alg::VUMPS{:C4v})
    @unpack AL, C, FL = rt
    AC = ALCtoAC_map(AL, C)
    _, FL = leftenv_c4v(AL, conj(AL), M, FL; alg)
    _, AC = ACenv_c4v(AC, FL, M; alg)
    _, C  = Cenv_c4v(C, FL; alg)

    QAC, RAC = qrpos(_to_front(AC))
    QC, RC = qrpos(C)
    AL = reshape(QAC*QC', size(AC))
    err = Zygote.@ignore norm(RAC - RC)

    return C4vVUMPSEnv(AL, C, FL), err
end

# ── Plaquette iteration + boundary ───────────────────────────────────

function vumps_itr(rt::C4vVUMPSEnv, M::StructArray, alg::VUMPS{:C4v})
    t = Zygote.@ignore time()
    atype = _arraytype(M)
    id = get_device_id(atype)
    local err

    Zygote.@ignore alg.verbosity >= 2 && @info "Start C4v VUMPS iteration without AD..."
    Zygote.@ignore for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            Zygote.@ignore @info @sprintf("C4vVUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 &&
                Zygote.@ignore @info @sprintf("C4vVUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("C4vVUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end

    Zygote.@ignore alg.verbosity >= 2 && @info "Start Plaquette VUMPS iteration with AD..."
    for i in 1:alg.maxiter_ad
        power_iter_backup = alg.power_iter
        alg.power_iter = alg.power_iter_ad
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step, rt, M, alg) : vumps_step(rt, M, alg)
        alg.power_iter = power_iter_backup
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("PlaqVUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("C4vVUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("C4vVUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end
    return rt, err
end

function leading_boundary(rt::C4vVUMPSEnv, M::StructArray, alg::VUMPS{:C4v})
    return vumps_itr(rt, M, alg)
end