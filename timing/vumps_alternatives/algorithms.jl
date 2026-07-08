# timing/vumps_alternatives/algorithms.jl
include(joinpath(@__DIR__, "common.jl"))

# One VUMPS outer step with plain power inner solves (mirror of c4v.jl vumps_step,
# but every map application counted). Returns (AL, C, FL, err).
function step_power(cnt, AL, C, FL, M; pi::Int)
    AC = TeneT.ALCtoAC_map(AL, C)
    for _ in 1:pi-1
        FL = flmap(cnt, FL, AL, M); FL /= norm(FL)
    end
    FL = flmap(cnt, FL, AL, M); FL /= norm(FL)
    for _ in 1:pi-1
        AC = acmap(cnt, AC, FL, M); AC /= norm(AC)
    end
    AC = acmap(cnt, AC, FL, M); AC /= norm(AC)
    for _ in 1:pi-1
        C = cmap(cnt, C, FL); C /= norm(C)
    end
    C = cmap(cnt, C, FL); C /= norm(C)
    AL, err = accto_al(AC, C)
    return AL, C, FL, err
end

function run_b0_power!(cnt, traj, S; pi, max_outer, tol, map_budget=10^9)
    (; M) = S
    AL, C, FL = S.rt.AL, S.rt.C, S.rt.FL
    t0 = time(); err = Inf
    for outer in 1:max_outer
        AL, C, FL, err = step_power(cnt, AL, C, FL, M; pi)
        f = diag_free_energy(cnt, AL, C, FL, M, S.model.beta)
        push_row!(traj; outer, fl=cnt.fl, ac=cnt.ac, c=cnt.c, nrm=cnt.nrm,
                  err, f, t=time()-t0)
        (err < tol || total_maps(cnt) > map_budget) && break
    end
    return (; AL, C, FL), err
end
