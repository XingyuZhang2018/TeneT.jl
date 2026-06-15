# cannon ObsEnv parity (4 ranks, 2×2 grid, CPU). Run: julia --project=. test/run_test_cannon_obsenv.jl
#
# Validates the distributed observation environment: serial ObsEnv vs cannon ObsEnv (which
# computes block obs envs via leftenv/rightenv_cannon ifobs=true, then gathers to full).
# Both single-env (VUMPSRuntime) and updown (Tuple) paths. Obs envs (FLo/FRo) are leftenv-
# ifobs eigenvectors (well-conditioned like FL) → compared up-to-gauge to machine precision.
# (The end-to-end physical check — cannon energy == serial == -0.4713 on a converged J1J2 A —
# is the stronger validation; it lives in a data-dependent script, not this regression gate.)
using Test, MPI, LinearAlgebra, Random, Printf, TeneT
using TeneT: cannon_grid, cannon_scatter, cannon_gather, VUMPS, General, Plaquette, Square,
             StructArray, VUMPSRuntime, PlaquetteVUMPSRuntime, ObsEnv
MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_obsenv.jl expects exactly 4 ranks"
say(s) = (rank == 0 && (println(s); flush(stdout)))

scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
scatter_rt(rt, g) = VUMPSRuntime(scatter_sa(rt.AL, g), scatter_sa(rt.AR, g), rt.C,
                                 scatter_sa(rt.FL, g), scatter_sa(rt.FR, g))   # C replicated
ph_relerr(a, b) = (c = dot(b, a) / dot(b, b); norm(a .- b .* c) / max(norm(b), eps()))
const FLDS = (:ACu, :ARu, :ACd, :ARd, :FLu, :FRu, :FLo, :FRo)

function build_rt(Ni, Nj, χ, D; seed, pat)
    Random.seed!(seed)
    nu = length(unique(pat))
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    rt = VUMPSRuntime(sa((χ,D,D,χ)), sa((χ,D,D,χ)), sa((χ,χ)), sa((χ,D,D,χ)), sa((χ,D,D,χ)))
    M  = StructArray([rand(ComplexF64, D, D, D, D, 2) for _ in 1:nu], pat)
    return rt, M
end
algp(g) = VUMPS{General}(grid=g, ifupdown=false, ifsimple_eig=true, forloop_iter=1,
                         power_iter=2, power_iter_obs=2, maxiter=1, maxiter_ad=1, verbosity=0)
cmp_env(env_s, env_c, nu) = maximum(maximum(ph_relerr(getfield(env_c, f).data[i], getfield(env_s, f).data[i])
                                            for i in 1:nu) for f in FLDS)

# ── Gate M5o-1: single-env ObsEnv parity ──────────────────────────────────────
@testset "Gate M5o-1: single-env ObsEnv parity" begin
    g = cannon_grid(2, 2)
    for (ci, (Ni, Nj, pat)) in enumerate([(1, 1, nothing), (2, 2, [1 3; 2 4])])
        χ, D = 14, 2
        rt, M = build_rt(Ni, Nj, χ, D; seed=4400 + ci, pat=(pat === nothing ? reshape(collect(1:Ni*Nj),Ni,Nj) : pat))
        alg_s = VUMPS{General}(grid=nothing, ifupdown=false, ifsimple_eig=true, forloop_iter=1,
                               power_iter=2, power_iter_obs=2, maxiter=1, maxiter_ad=1, verbosity=0)
        env_s = ObsEnv(rt, M, alg_s, nothing)
        env_c = ObsEnv(scatter_rt(rt, g), M, algp(g), nothing)   # cannon → gathered full VUMPSEnv
        e = cmp_env(env_s, env_c, length(rt.AL.data))
        say(@sprintf("  [M5o-1 case %d %dx%d] ObsEnv max ph_relerr = %.1e", ci, Ni, Nj, e))
        @test e ≤ 1e-7
    end
end

# ── Gate M5o-2: updown ObsEnv parity (Tuple) ──────────────────────────────────
@testset "Gate M5o-2: updown ObsEnv parity" begin
    g = cannon_grid(2, 2)
    χ, D = 14, 2
    pat = [1 3; 2 4]
    rtup, M  = build_rt(2, 2, χ, D; seed=5500, pat=pat)
    rtdown, _ = build_rt(2, 2, χ, D; seed=5501, pat=pat)
    alg_s = VUMPS{General}(grid=nothing, ifupdown=true, ifdownfromup=false, ifparallelupdown=false,
                           ifsimple_eig=true, forloop_iter=1, power_iter=2, power_iter_obs=2,
                           maxiter=1, maxiter_ad=1, verbosity=0)
    alg_c = deepcopy(alg_s); alg_c.grid = g
    env_s = ObsEnv((rtup, rtdown), M, alg_s, nothing)
    env_c = ObsEnv((scatter_rt(rtup, g), scatter_rt(rtdown, g)), M, alg_c, nothing)
    e = cmp_env(env_s, env_c, length(rtup.AL.data))
    say(@sprintf("  [M5o-2 updown] ObsEnv max ph_relerr = %.1e", e))
    @test e ≤ 1e-7
end

# ── Gate M5o-3: Plaquette ObsEnv parity ───────────────────────────────────────
@testset "Gate M5o-3: Plaquette ObsEnv parity" begin
    g = cannon_grid(2, 2)
    χ, D = 14, 2; pat = [1 3; 2 4]; nu = 4
    Random.seed!(6600)
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    rt = PlaquetteVUMPSRuntime(sa((χ,D,D,χ)), sa((χ,χ)), sa((χ,D,D,χ)))
    M  = StructArray([rand(ComplexF64, D, D, D, D, 2) for _ in 1:nu], pat)
    alg_s = VUMPS{Plaquette{Square}}(grid=nothing, ifsimple_eig=true, ifupdown=false,
                                     power_iter=2, power_iter_obs=2, forloop_iter=1,
                                     maxiter=1, maxiter_ad=1, verbosity=0)
    alg_c = deepcopy(alg_s); alg_c.grid = g
    rtb = PlaquetteVUMPSRuntime(scatter_sa(rt.AL, g), rt.C, scatter_sa(rt.FL, g))
    env_s = ObsEnv(rt, M, alg_s, nothing)
    env_c = ObsEnv(rtb, M, alg_c, nothing)   # cannon → gathered full PlaquetteVUMPSEnv
    e = maximum(maximum(ph_relerr(getfield(env_c, f).data[i], getfield(env_s, f).data[i]) for i in 1:nu)
                for f in (:AL, :C, :FLu, :FLo))
    say(@sprintf("  [M5o-3 Plaquette] ObsEnv max ph_relerr = %.1e", e))
    @test e ≤ 1e-7
end

say("all cannon-ObsEnv gates done.")
