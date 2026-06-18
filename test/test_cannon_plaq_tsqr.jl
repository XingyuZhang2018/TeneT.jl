# Plaquette Cannon forward-only TSQR seam (4 ranks, 2x2 grid, CPU).
# Run via: julia --project=. test/run_test_cannon_plaq_tsqr.jl
using Test, MPI, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: cannon_grid, cannon_scatter, cannon_gather, split_ranges,
             VUMPS, Plaquette, Square, StructArray,
             init_VUMPSRuntime_cannon, ACCtoAL, ACCtoAL_tsqr_cannon

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_plaq_tsqr.jl expects exactly 4 ranks"

scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g)  = StructArray([cannon_gather(t, g)  for t in SA.data], SA.pattern)
to_front(t) = reshape(t, Int(prod(size(t)) / size(t, 1)), size(t, 1))
relerr(a, b) = norm(a .- b) / max(norm(b), eps(real(eltype(b))))

const PATS = [[1 3; 2 4], [1 2; 2 1]]

@testset "Plaquette ACCtoAL TSQR seam parity" begin
    g = cannon_grid(2, 2)
    for (ci, pat) in enumerate(PATS)
        chi, D = 16, 2
        nu = length(unique(pat))
        Random.seed!(1100 + ci)
        AC = StructArray([rand(ComplexF64, chi, D, D, chi) for _ in 1:nu], pat)
        C = StructArray([Matrix(qr(rand(ComplexF64, chi, chi)).Q) for _ in 1:nu], pat)

        ALs, err_s = ACCtoAL(AC, C)
        ALt, err_t = ACCtoAL_tsqr_cannon(scatter_sa(AC, g), C, g)
        ALtf = gather_sa(ALt, g)

        max_al = maximum(relerr(ALtf.data[i], ALs.data[i]) for i in 1:nu)
        rank == 0 && println("  TSQR seam case $ci max AL relerr = $max_al; err diff = $(abs(err_t - err_s))")
        @test max_al <= 1e-10
        @test abs(err_t - err_s) <= 1e-10
    end
end

@testset "Plaquette ACCtoAL TSQR seam gradient parity" begin
    g = cannon_grid(2, 2)
    chi, D = 10, 2
    a_rs = split_ranges(chi, g.N1)
    l_rs = split_ranges(chi, g.N2)
    blkof(t) = t[a_rs[g.r1 + 1], :, :, l_rs[g.r2 + 1]]
    for (ci, pat) in enumerate(PATS)
        nu = length(unique(pat))
        Random.seed!(1300 + ci)
        AC = StructArray([rand(ComplexF64, chi, D, D, chi) for _ in 1:nu], pat)
        C = StructArray([Matrix(qr(rand(ComplexF64, chi, chi)).Q) for _ in 1:nu], pat)
        ACb = scatter_sa(AC, g)
        Random.seed!(1350 + ci)
        WAL = [rand(ComplexF64, chi, D, D, chi) for _ in 1:nu]
        WALb = [cannon_scatter(W, g) for W in WAL]

        lserial(ac, c) = let (al, _) = ACCtoAL(ac, c)
            real(sum(sum(conj(WAL[k]) .* al.data[k]) for k in 1:nu))
        end
        ltsqr(ac, c) = let (al, _) = ACCtoAL_tsqr_cannon(ac, c, g)
            real(sum(sum(conj(WALb[k]) .* al.data[k]) for k in 1:nu))
        end

        gr = Zygote.gradient(lserial, AC, C)
        gt = Zygote.gradient(ltsqr, ACb, C)
        eAC = maximum(norm(gt[1].data[k] - blkof(gr[1].data[k])) / max(norm(blkof(gr[1].data[k])), 1e-12) for k in 1:nu)
        eC = maximum(norm(gt[2].data[k] - gr[2].data[k]) / max(norm(gr[2].data[k]), 1e-12) for k in 1:nu)
        rank == 0 && println("  TSQR seam gradient case $ci dAC = $eAC; dC = $eC")
        @test eAC <= 1e-7
        @test eC <= 1e-7
    end
end

@testset "Plaquette distributed_qr init sanity" begin
    g = cannon_grid(2, 2)
    chi, D = 12, 2
    pat = [1 3; 2 4]
    Random.seed!(1200)
    M = StructArray([rand(ComplexF64, D, D, D, D, 2) for _ in 1:4], pat)

    # Diverge the ambient RNG; distributed init must be rank-uniform anyway.
    for _ in 1:rank
        rand(ComplexF64)
    end

    alg = VUMPS{Plaquette{Square}}(grid=g, distributed_qr=true, ifsimple_eig=true,
                                   ifupdown=false, power_iter=1, forloop_iter=1,
                                   maxiter=1, maxiter_ad=0, verbosity=0)
    rt = init_VUMPSRuntime_cannon(M, chi, g, alg)
    ALf = gather_sa(rt.AL, g)
    FLf = gather_sa(rt.FL, g)

    for (SA, name) in ((ALf, "AL"), (FLf, "FL"), (rt.C, "C"))
        for i in 1:length(SA.data)
            ref = MPI.bcast(SA.data[i], 0, comm)
            @test maximum(abs, SA.data[i] .- ref) <= 1e-12
        end
        rank == 0 && println("  distributed_qr init $name is rank-uniform")
    end

    for A in ALf.data
        Q = to_front(A)
        @test norm(Q' * Q - I) / chi <= 1e-10
    end
end

rank == 0 && println("all Plaquette TSQR gates done.")
