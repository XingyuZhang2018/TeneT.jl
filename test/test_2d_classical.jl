function _expected_ising_mpo(beta)
    B = [exp(beta) exp(-beta);
         exp(-beta) exp(beta)]
    W = sqrt(B)
    M = zeros(2, 2, 2, 2)
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2, s in 1:2
        M[a, b, c, d] += W[s, a] * W[s, b] * W[s, c] * W[s, d]
    end
    return M
end

function _expected_exact_free_energy(beta; npts)
    s = 0.0
    dt = pi / npts
    for i in 1:npts
        t1 = (i - 0.5) * dt
        for j in 1:npts
            t2 = (j - 0.5) * dt
            s += log(cosh(2 * beta)^2 - sinh(2 * beta) * (cos(t1) + cos(t2)))
        end
    end
    s *= dt^2 / (2 * pi^2)
    return -(log(2) + s) / beta
end

@testset "2D classical models" begin
    @testset "Ising square MPO helpers" begin
        beta = 0.3
        model = Ising(lattice=Square(), beta=beta)

        M = MPO(model; atype=Array)
        @test M isa TeneT.StructArray
        @test size(M) == (1, 1)
        @test size(M[1]) == (2, 2, 2, 2)
        @test M.pattern == [1;;]
        @test M[1] ≈ _expected_ising_mpo(beta)

        M_c4v = MPO(model, C4v; atype=Array)
        @test M_c4v isa TeneT.StructArray
        @test size(M_c4v) == (1, 1)
        @test size(M_c4v[1]) == (2, 2, 2, 2, 1)
        @test M_c4v[1][:, :, :, :, 1] ≈ M[1]
        @test !isdefined(TeneT, :ising_mpo)
        @test !isdefined(TeneT, :ising_c4v_mpo)
    end

    @testset "Ising free energy helpers" begin
        beta = 0.3
        model = Ising(lattice=Square(), beta=beta)

        f_exact = exact_free_energy(model; npts=256)
        @test f_exact ≈ _expected_exact_free_energy(beta; npts=256)

        z_per_site = exp(-beta * f_exact)
        @test free_energy(model, z_per_site) ≈ f_exact
        @test free_energy(model, z_per_site^2; log_power=2) ≈ f_exact
    end
end
