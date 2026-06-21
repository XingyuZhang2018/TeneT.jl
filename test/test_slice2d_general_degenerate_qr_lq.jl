using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: slice2d_grid, slice2d_scatter, split_ranges,
             StructArray, ACCtoALAR, ACCtoALAR_dist_slice2d

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 8 "test_slice2d_general_degenerate_qr_lq.jl expects exactly 8 ranks"

say(s) = (rank == 0 && (println(s); flush(stdout)))

scatter_sa(SA, g) = StructArray([slice2d_scatter(t, g) for t in SA.data], SA.pattern)

function build_inputs(chi, D, pat; seed)
    Random.seed!(seed)
    nu = length(unique(pat))
    AC = StructArray([rand(ComplexF64, chi, D, D, chi) for _ in 1:nu], pat)
    C = StructArray([Matrix(qr(rand(ComplexF64, chi, chi)).Q) for _ in 1:nu], pat)
    return AC, C
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

@testset "General distributed QR/LQ gradients on degenerate 8-rank grids" begin
    chi, D = 16, 2
    pat = reshape(collect(1:4), 2, 2)
    for (case, N1, N2) in (("8x1", 8, 1), ("1x8", 1, 8))
        say("case $case")
        g = slice2d_grid(N1, N2)
        AC, C = build_inputs(chi, D, pat; seed=5100 + 10N1 + N2)
        ACb = scatter_sa(AC, g)

        Random.seed!(5200 + 10N1 + N2)
        WAL = [rand(ComplexF64, chi, D, D, chi) for _ in 1:length(AC.data)]
        WAR = [rand(ComplexF64, chi, D, D, chi) for _ in 1:length(AC.data)]
        WALb = [slice2d_scatter(w, g) for w in WAL]
        WARb = [slice2d_scatter(w, g) for w in WAR]

        loss_ref(ac, c) = let (al, ar, _, _) = ACCtoALAR(ac, c)
            real(sum(sum(conj(WAL[k]) .* al.data[k]) + sum(conj(WAR[k]) .* ar.data[k])
                     for k in eachindex(WAL)))
        end
        loss_dist(ac, c) = let (al, ar, _, _) = ACCtoALAR_dist_slice2d(ac, c, g)
            real(sum(sum(conj(WALb[k]) .* al.data[k]) + sum(conj(WARb[k]) .* ar.data[k])
                     for k in eachindex(WALb)))
        end

        gr = Zygote.gradient(loss_ref, AC, C)
        gd = Zygote.gradient(loss_dist, ACb, C)
        assert_block_grad("$case ACCtoALAR dAC", gd[1], gr[1], g)
        for k in eachindex(C.data)
            @test isapprox(gd[2].data[k], gr[2].data[k]; rtol=1e-7, atol=1e-10)
        end
    end
end

say("all degenerate General distributed QR/LQ gates done.")
