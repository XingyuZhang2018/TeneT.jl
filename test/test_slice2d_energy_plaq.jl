# Distributed energy_value parity — Plaquette J1J2 Square (4 ranks, 2×2 grid, CPU).
# Run: julia --project=. test/run_test_slice2d_energy_plaq.jl
#
# Serial energy_value (grid=nothing, full env) vs slice2d energy_value (grid set, BLOCK env
# from the no-gather ObsEnv path, contractions via FLmap/ACmap/FRmap/ACdmap_slice2d_dist +
# slice2d_dot). Same random iPEPS A + boundary runtime; compares the per-bond e_dict and the
# total energy. J1H/J1V/diag(imag) kernels are exact → machine precision; J2 (oc_22) carries
# the ~1e-10 random-Q0 noise floor (serial AND slice2d each have an independent realization),
# so it is gated looser (≤1e-8) — a wrong contraction would be O(1), not 1e-10.
using Test, MPI, LinearAlgebra, Random, Printf, TeneT, Zygote
using TeneT: slice2d_grid, slice2d_scatter, VUMPS, Plaquette, Square, J1J2, Heisenberg, StructArray,
             PlaquetteVUMPSRuntime, ObsEnv, energy_value, imag_error, GradientOptimize,
             _imag_error_op, contract_n_11, contract_n_12, contract_n_21, contract_n_22
MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_slice2d_energy_plaq.jl expects exactly 4 ranks"
say(s) = (rank == 0 && (println(s); flush(stdout)))

scatter_sa(SA, g) = StructArray([slice2d_scatter(t, g) for t in SA.data], SA.pattern)
scatter_rt(rt, g) = PlaquetteVUMPSRuntime(scatter_sa(rt.AL, g), rt.C, scatter_sa(rt.FL, g)) # C replicated

# max |serial - slice2d| over a bond-energy sub-dict
dict_maxdiff(ds, dc) = maximum(abs(ds[k] - dc[k]) for k in keys(ds))

@testset "Gate M6: distributed energy_value parity (Plaquette J1J2 Square)" begin
    g = slice2d_grid(2, 2)
    χ, D, d = 14, 2, 2
    pat = [1 3; 2 4]; nu = 4
    Random.seed!(7700)
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    A  = sa((D, D, D, D, d))                                   # iPEPS = M (leg5)
    rt = PlaquetteVUMPSRuntime(sa((χ,D,D,χ)), sa((χ,χ)), sa((χ,D,D,χ)))   # AL, C, FL

    model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.6,
                 ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    mkalg(grid) = VUMPS{Plaquette{Square}}(grid=grid, ifsimple_eig=true, ifparallel=false,
                                           forloop_iter=1, power_iter=2, power_iter_obs=2,
                                           maxiter=1, maxiter_ad=1, verbosity=0)
    alg_s = mkalg(nothing); alg_c = mkalg(g)
    params_s = GradientOptimize(model=model, pattern=pat, boundary_alg=alg_s)
    params_c = GradientOptimize(model=model, pattern=pat, boundary_alg=alg_c)

    env_s = ObsEnv(rt,              A, alg_s, model)   # full PlaquetteVUMPSEnv
    env_c = ObsEnv(scatter_rt(rt,g), A, alg_c, model)  # BLOCK PlaquetteVUMPSEnv (J1J2{Square})

    E_s, ed_s = energy_value(model, A, env_s, params_s)
    E_c, ed_c = energy_value(model, A, env_c, params_c)

    eJ1H = dict_maxdiff(ed_s["bond_J1H_energy"], ed_c["bond_J1H_energy"])
    eJ1V = dict_maxdiff(ed_s["bond_J1V_energy"], ed_c["bond_J1V_energy"])
    eJ2  = dict_maxdiff(ed_s["bond_J2\\_energy"], ed_c["bond_J2\\_energy"])
    eTot = abs(E_s - E_c)
    say(@sprintf("  [M6 energy] E_serial=%.12g%+.3eim  E_slice2d=%.12g%+.3eim",
                 real(E_s), imag(E_s), real(E_c), imag(E_c)))
    say(@sprintf("  [M6 energy] maxdiff  J1H=%.2e  J1V=%.2e  J2=%.2e  total=%.2e",
                 eJ1H, eJ1V, eJ2, eTot))
    @test eJ1H ≤ 1e-9
    @test eJ1V ≤ 1e-9
    @test eJ2  ≤ 1e-8
    @test eTot ≤ 1e-8

    # imag_error (single oc_11; quadratic in FLo → gauge-invariant): exact parity
    iSy = _imag_error_op(A, params_s)
    ie_s = imag_error(env_s, A, iSy, params_s)
    ie_c = imag_error(env_c, A, iSy, params_c)
    say(@sprintf("  [M6 imag_error] serial=%.12g slice2d=%.12g diff=%.2e", ie_s, ie_c, abs(ie_s-ie_c)))
    @test abs(ie_s - ie_c) ≤ 1e-9
end

# ── Gate M6-grad: gradient parity dE/dA (the optimization-critical check) ──────
# energy_value is differentiated in energy(); the slice2d contraction's A-gradient must match
# serial. A is replicated → dA must be the FULL gradient on every rank. Two-level check:
#  (1) Per-kernel: oc_11/12/21 are deterministic (no random Q0) → exact machine precision,
#      which directly validates the FLmap/ACmap_slice2d_dist + slice2d_dot backward rrules.
#  (2) Full energy_value: oc_22's value is Q0-independent but its GRADIENT carries the
#      random-Q0 sensitivity SERIAL has too — the qrpos backward is ill-conditioned for a
#      poorly-converged obs env, amplifying both slice2d's FP-reassociation and serial's Q0
#      jitter to the same floor. So the rigorous statement is: slice2d adds no more gradient
#      error than re-running serial with a fresh Q0 (slice2d-vs-serial ≤ serial-vs-serial).
gflat(g) = reduce(vcat, vec.(g.data))   # StructArray / Tangent both expose .data
@testset "Gate M6-grad: per-kernel dE/dA (deterministic kernels exact)" begin
    g = slice2d_grid(2, 2); χ, D, d = 14, 2, 2
    Random.seed!(8800)
    mk(dims...) = rand(ComplexF64, dims...); sc(t) = slice2d_scatter(t, g)
    FLo=mk(χ,D,D,χ); FLu=mk(χ,D,D,χ); ACu=mk(χ,D,D,χ); ACd=mk(χ,D,D,χ)
    FRo=mk(χ,D,D,χ); FRu=mk(χ,D,D,χ); ARu=mk(χ,D,D,χ); ARd=mk(χ,D,D,χ); A=mk(D,D,D,D,d)
    relg(gs,gc) = norm(gs-gc)/max(norm(gs),eps())
    import TeneT: contract_n_11, contract_n_12, contract_n_21
    g11 = relg(Zygote.gradient(a->real(contract_n_11(FLo,ACu,a,ACd,FRo; forloop_iter=1,ifparallel=false,grid=nothing)),A)[1],
               Zygote.gradient(a->real(contract_n_11(sc(FLo),sc(ACu),a,sc(ACd),sc(FRo); forloop_iter=1,ifparallel=false,grid=g)),A)[1])
    g12 = relg(Zygote.gradient(a->real(contract_n_12(FLo,ACu,a,ACd,FRo,ARu,a,ARd; forloop_iter=1,ifparallel=false,grid=nothing)),A)[1],
               Zygote.gradient(a->real(contract_n_12(sc(FLo),sc(ACu),a,sc(ACd),sc(FRo),sc(ARu),a,sc(ARd); forloop_iter=1,ifparallel=false,grid=g)),A)[1])
    g21 = relg(Zygote.gradient(a->real(contract_n_21(ACu,FLu,a,FRu,FLo,a,FRo,ACd; forloop_iter=1,ifparallel=false,grid=nothing)),A)[1],
               Zygote.gradient(a->real(contract_n_21(sc(ACu),sc(FLu),a,sc(FRu),sc(FLo),a,sc(FRo),sc(ACd); forloop_iter=1,ifparallel=false,grid=g)),A)[1])
    say(@sprintf("  [M6-grad kernels] oc_11=%.2e oc_12=%.2e oc_21=%.2e", g11, g12, g21))
    @test g11 ≤ 1e-10
    @test g12 ≤ 1e-10
    @test g21 ≤ 1e-10

    # oc_22 (qrpos seam): on a WELL-conditioned random env the noise floor is ~1e-9 (NOT the
    # 5e-4 of the ill-conditioned obs env), so this is a DISCRIMINATING check — slice2d must track
    # serial's inherent Q0 noise down here. Assert slice2d-vs-serial ≤ 2× serial-vs-serial (fresh
    # Q0), which a fixed-size slice2d bug ≳1e-7 could not satisfy at this low floor.
    f22s(a)=real(contract_n_22(FLu,FLo,ACu,ACd,FRu,FRo,ARu,ARd,a,a,a,a; forloop_iter=1,ifparallel=false,grid=nothing))
    f22c(a)=real(contract_n_22(sc(FLu),sc(FLo),sc(ACu),sc(ACd),sc(FRu),sc(FRo),sc(ARu),sc(ARd),a,a,a,a; forloop_iter=1,ifparallel=false,grid=g))
    g22s=Zygote.gradient(f22s,A)[1]; g22c=Zygote.gradient(f22c,A)[1]; g22s2=Zygote.gradient(f22s,A)[1]
    g22  = relg(g22s, g22c)        # slice2d vs serial (independent Q0)
    g22n = relg(g22s, g22s2)       # serial vs serial (independent Q0) — the inherent floor
    say(@sprintf("  [M6-grad kernels] oc_22 slice2d-vs-serial=%.2e  serial-vs-serial=%.2e", g22, g22n))
    @test g22 ≤ max(2 * g22n, 1e-7)
end

@testset "Gate M6-grad: full dE/dA — slice2d ≤ serial's inherent Q0 noise" begin
    g = slice2d_grid(2, 2)
    χ, D, d = 14, 2, 2
    pat = [1 3; 2 4]; nu = 4
    Random.seed!(7711)
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    A  = sa((D, D, D, D, d))
    rt = PlaquetteVUMPSRuntime(sa((χ,D,D,χ)), sa((χ,χ)), sa((χ,D,D,χ)))
    model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.6,
                 ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    mkalg(grid) = VUMPS{Plaquette{Square}}(grid=grid, ifsimple_eig=true, ifparallel=false,
                                           forloop_iter=1, power_iter=2, power_iter_obs=2,
                                           maxiter=1, maxiter_ad=1, verbosity=0)
    alg_s = mkalg(nothing); alg_c = mkalg(g)
    params_s = GradientOptimize(model=model, pattern=pat, boundary_alg=alg_s)
    params_c = GradientOptimize(model=model, pattern=pat, boundary_alg=alg_c)
    env_s = ObsEnv(rt,              A, alg_s, model)
    env_c = ObsEnv(scatter_rt(rt,g), A, alg_c, model)

    g_s  = Zygote.gradient(a -> real(energy_value(model, a, env_s, params_s)[1]), A)[1]
    g_c  = Zygote.gradient(a -> real(energy_value(model, a, env_c, params_c)[1]), A)[1]
    g_s2 = Zygote.gradient(a -> real(energy_value(model, a, env_s, params_s)[1]), A)[1]  # fresh Q0
    rel    = norm(gflat(g_s) - gflat(g_c )) / max(norm(gflat(g_s)), eps())
    rel_ss = norm(gflat(g_s) - gflat(g_s2)) / max(norm(gflat(g_s)), eps())
    say(@sprintf("  [M6-grad full] serial-vs-slice2d = %.2e   serial-vs-serial (fresh Q0) = %.2e", rel, rel_ss))
    @test rel ≤ max(3 * rel_ss, 1e-7)   # slice2d no worse than serial's own Q0 jitter
end

# ── Gate M6-scope: ObsEnv block-return is gated to J1J2{Square} only (B2 regression) ──────────
# A non-J1J2{Square} Plaquette model with grid set MUST still gather to a FULL env (its
# energy_value — e.g. Honeycomb's oc_13, Heisenberg — is not slice2d-ized). Block χ = χ/N1; a
# full env keeps χ. This guards against a future widening of the _dist_energy_plaq predicate.
@testset "Gate M6-scope: ObsEnv no-gather is J1J2{Square}-only" begin
    g = slice2d_grid(2, 2); χ, D, d = 14, 2, 2; pat = [1 3; 2 4]; nu = 4
    Random.seed!(7733)
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    A  = sa((D, D, D, D, d))
    rt = PlaquetteVUMPSRuntime(sa((χ,D,D,χ)), sa((χ,χ)), sa((χ,D,D,χ)))
    alg_c = VUMPS{Plaquette{Square}}(grid=g, ifsimple_eig=true, ifparallel=false, forloop_iter=1,
                                     power_iter=2, power_iter_obs=2, maxiter=1, maxiter_ad=1, verbosity=0)
    # J1J2{Square}: block (FLo first leg = χ/N1)
    env_sq = ObsEnv(scatter_rt(rt,g), A, alg_c,
                    J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.6, ifrotate=true, couplingtype=:uniform, bondratio=1.0))
    # Heisenberg{Square}: NOT slice2d-ized → must gather to FULL (FLo first leg = χ)
    env_he = ObsEnv(scatter_rt(rt,g), A, alg_c, Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0))
    say(@sprintf("  [M6-scope] FLo first-dim: J1J2{Square}=%d (block, χ/N1=%d)  Heisenberg=%d (full, χ=%d)",
                 size(env_sq.FLo[1,1],1), χ ÷ 2, size(env_he.FLo[1,1],1), χ))
    @test size(env_sq.FLo[1,1], 1) < χ      # block
    @test size(env_he.FLo[1,1], 1) == χ     # gathered to full (B2 preserved)
end

say("all slice2d distributed-energy gates done.")
