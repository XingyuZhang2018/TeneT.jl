using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: cannon_grid, cannon_scatter, cannon_gather, split_ranges,
             VUMPS, General, StructArray, VUMPSRuntime, Recompute,
             ALCtoAC, ALCtoAC_cannon,
             ACCtoAL, ACCtoAL_tsqr_cannon,
             ACCtoAR, ACCtoAR_tslq_cannon,
             ACCtoALAR, ACCtoALAR_dist_cannon,
             vumps_step, vumps_step_cannon, checkpoint

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_general_dist_qr_lq.jl expects exactly 4 ranks"
say(s) = (rank == 0 && (println(s); flush(stdout)))

scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g) = StructArray([cannon_gather(t, g) for t in SA.data], SA.pattern)
scatter_rt(rt, g) = VUMPSRuntime(scatter_sa(rt.AL, g), scatter_sa(rt.AR, g), rt.C,
                                 scatter_sa(rt.FL, g), scatter_sa(rt.FR, g))
ph_relerr(a, b) = (c = dot(b, a) / dot(b, b); norm(a .- b .* c) / max(norm(b), eps()))

function build_inputs(χ, D, pat; seed)
    Random.seed!(seed)
    nu = length(unique(pat))
    AC = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], pat)
    AL = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], pat)
    C = StructArray([Matrix(qr(rand(ComplexF64, χ, χ)).Q) for _ in 1:nu], pat)
    return AL, AC, C
end

function block_of(t, g)
    a_rs = split_ranges(size(t, 1), g.N1)
    l_rs = split_ranges(size(t, ndims(t)), g.N2)
    inds = ntuple(i -> i == 1 ? a_rs[g.r1 + 1] :
                       (i == ndims(t) ? l_rs[g.r2 + 1] : Colon()), ndims(t))
    return t[inds...]
end

function assert_block_grad(label, got, ref, g; rtol=1e-7, atol=1e-10)
    @test length(got.data) == length(ref.data)
    for k in eachindex(ref.data)
        rb = block_of(ref.data[k], g)
        err = norm(got.data[k] - rb) / max(norm(rb), atol)
        rank == 0 && println("  $label block $k relerr = $err")
        @test isapprox(got.data[k], rb; rtol, atol)
    end
end

function general_vumps_step_body()
    cannon_src = read(joinpath(@__DIR__, "..", "src", "boundary_algorithm", "vumps", "cannon.jl"), String)
    sig = "function vumps_step_cannon(rt::VUMPSRuntime, M::StructArray, grid::CannonGrid, alg::VUMPS{General})"
    start = findfirst(sig, cannon_src)
    @test start !== nothing
    tail = cannon_src[last(start):end]
    marker = "# Fixed seed for the distributed init"
    stop = findfirst(marker, tail)
    @test stop !== nothing
    return tail[1:first(stop)-1]
end

const PATS = (reshape(collect(1:4), 2, 2), [1 2; 2 1])

@testset "General distributed ALCtoAC forward and gradient parity" begin
    g = cannon_grid(2, 2)
    χ, D = 10, 2
    for (ci, pat) in enumerate(PATS)
        AL, _, C = build_inputs(χ, D, pat; seed=1100 + ci)
        ALb = scatter_sa(AL, g)
        ACs = ALCtoAC(AL, C)
        ACc = ALCtoAC_cannon(ALb, C, g)
        ACcf = gather_sa(ACc, g)
        @test maximum(norm(ACcf.data[k] - ACs.data[k]) for k in eachindex(ACs.data)) <= 1e-10

        Random.seed!(1150 + ci)
        W = [rand(ComplexF64, χ, D, D, χ) for _ in 1:length(ACs.data)]
        Wb = [cannon_scatter(w, g) for w in W]
        loss_ref(al, c) = real(sum(sum(conj(W[k]) .* ALCtoAC(al, c).data[k]) for k in eachindex(W)))
        loss_can(al, c) = real(sum(sum(conj(Wb[k]) .* ALCtoAC_cannon(al, c, g).data[k]) for k in eachindex(Wb)))
        gr = Zygote.gradient(loss_ref, AL, C)
        gc = Zygote.gradient(loss_can, ALb, C)
        assert_block_grad("ALCtoAC dAL", gc[1], gr[1], g)
        for k in eachindex(C.data)
            @test isapprox(gc[2].data[k], gr[2].data[k]; rtol=1e-7, atol=1e-10)
        end
    end
end

@testset "row first-dimension gather primitive parity" begin
    g = cannon_grid(2, 2)
    Random.seed!(1001 + rank)
    local_rows = g.r2 == 0 ? 3 : 2
    blk = rand(ComplexF64, local_rows, 4)
    rs = split_ranges(5, g.N2)
    full = TeneT.cannon_gather_first_row(blk, g, rs)
    @test size(full) == (5, 4)
    @test full[rs[g.r2 + 1], :] == blk

    W = rand(ComplexF64, 5, 4)
    loss(x) = real(sum(conj(W) .* TeneT.cannon_gather_first_row(x, g, rs)))
    gb = Zygote.gradient(loss, blk)[1]
    expected = TeneT._cannon_row_reduce_scatter_first(W, g, rs)
    @test isapprox(gb, expected; rtol=1e-12, atol=1e-12)
end

@testset "General distributed QR/LQ forward parity" begin
    g = cannon_grid(2, 2)
    χ, D = 12, 2
    for (ci, pat) in enumerate(PATS)
        _, AC, C = build_inputs(χ, D, pat; seed=1200 + ci)
        ACb = scatter_sa(AC, g)

        ALs, errLs = ACCtoAL(AC, C)
        ALc, errLc = ACCtoAL_tsqr_cannon(ACb, C, g)
        ALcf = gather_sa(ALc, g)
        @test maximum(norm(ALcf.data[k] - ALs.data[k]) / max(norm(ALs.data[k]), 1e-12) for k in eachindex(ALs.data)) <= 1e-10
        @test abs(errLc - errLs) <= 1e-10

        ARs, errRs = ACCtoAR(AC, C)
        ARc, errRc = ACCtoAR_tslq_cannon(ACb, C, g)
        ARcf = gather_sa(ARc, g)
        @test maximum(norm(ARcf.data[k] - ARs.data[k]) / max(norm(ARs.data[k]), 1e-12) for k in eachindex(ARs.data)) <= 1e-10
        @test abs(errRc - errRs) <= 1e-10

        AL2, AR2, eL2, eR2 = ACCtoALAR_dist_cannon(ACb, C, g)
        AL2f = gather_sa(AL2, g)
        AR2f = gather_sa(AR2, g)
        @test maximum(norm(AL2f.data[k] - ALcf.data[k]) / max(norm(ALcf.data[k]), 1e-12) for k in eachindex(ALcf.data)) <= 1e-10
        @test maximum(norm(AR2f.data[k] - ARcf.data[k]) / max(norm(ARcf.data[k]), 1e-12) for k in eachindex(ARcf.data)) <= 1e-10
        @test isapprox(eL2, errLc; rtol=1e-10, atol=1e-10)
        @test isapprox(eR2, errRc; rtol=1e-10, atol=1e-10)
    end
end

@testset "General distributed QR/LQ gradient parity" begin
    g = cannon_grid(2, 2)
    χ, D = 10, 2
    for (ci, pat) in enumerate(PATS)
        _, AC, C = build_inputs(χ, D, pat; seed=1300 + ci)
        ACb = scatter_sa(AC, g)
        Random.seed!(1350 + ci)
        WAL = [rand(ComplexF64, χ, D, D, χ) for _ in 1:length(AC.data)]
        WAR = [rand(ComplexF64, χ, D, D, χ) for _ in 1:length(AC.data)]
        WALb = [cannon_scatter(w, g) for w in WAL]
        WARb = [cannon_scatter(w, g) for w in WAR]

        loss_ref(ac, c) = let (al, ar, _, _) = ACCtoALAR(ac, c)
            real(sum(sum(conj(WAL[k]) .* al.data[k]) + sum(conj(WAR[k]) .* ar.data[k]) for k in eachindex(WAL)))
        end
        loss_can(ac, c) = let (al, ar, _, _) = ACCtoALAR_dist_cannon(ac, c, g)
            real(sum(sum(conj(WALb[k]) .* al.data[k]) + sum(conj(WARb[k]) .* ar.data[k]) for k in eachindex(WALb)))
        end
        gr = Zygote.gradient(loss_ref, AC, C)
        gc = Zygote.gradient(loss_can, ACb, C)
        assert_block_grad("ACCtoALAR dAC", gc[1], gr[1], g)
        for k in eachindex(C.data)
            @test isapprox(gc[2].data[k], gr[2].data[k]; rtol=1e-7, atol=1e-10)
        end

        loss_recompute(ac, c) = let (al, ar, _, _) = checkpoint(Recompute(), ACCtoALAR_dist_cannon, ac, c, g)
            real(sum(sum(conj(WALb[k]) .* al.data[k]) + sum(conj(WARb[k]) .* ar.data[k]) for k in eachindex(WALb)))
        end
        gcr = Zygote.gradient(loss_recompute, ACb, C)
        assert_block_grad("ACCtoALAR recompute dAC", gcr[1], gr[1], g)
    end
end

@testset "General vumps_step_cannon uses distributed seams" begin
    step_body = general_vumps_step_body()
    @test occursin("ACCtoALAR_dist_cannon", step_body)
    @test !occursin("ACCtoALAR_cannon(ac, c, grid)", step_body)
    g = cannon_grid(2, 2)
    χ, D = 8, 2
    pat = reshape(collect(1:4), 2, 2)
    Random.seed!(1400)
    nu = length(unique(pat))
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    rt = VUMPSRuntime(sa((χ, D, D, χ)), sa((χ, D, D, χ)), sa((χ, χ)), sa((χ, D, D, χ)), sa((χ, D, D, χ)))
    M = StructArray([rand(ComplexF64, D, D, D, D, 2) for _ in 1:nu], pat)
    alg_s = VUMPS(General(); ifsimple_eig=true, ifupdown=false, power_iter=2, forloop_iter=1, maxiter=1, maxiter_ad=1, verbosity=0)
    alg_c = VUMPS(General(); ifsimple_eig=true, ifupdown=false, power_iter=2, forloop_iter=1, maxiter=1, maxiter_ad=1, verbosity=0, grid=g)
    rtb = scatter_rt(rt, g)
    rt_s, err_s = vumps_step(rt, M, alg_s)
    rt_c, err_c = vumps_step_cannon(rtb, M, g, alg_c)
    @test abs(err_c - err_s) <= 1e-8
    @test maximum(ph_relerr(gather_sa(rt_c.FL, g).data[k], rt_s.FL.data[k]) for k in eachindex(rt_s.FL.data)) <= 1e-8
    @test maximum(ph_relerr(gather_sa(rt_c.FR, g).data[k], rt_s.FR.data[k]) for k in eachindex(rt_s.FR.data)) <= 1e-8
end

say("all General distributed QR/LQ gates done.")
