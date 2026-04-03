@testset "Clifford group" begin

    @testset "Single-qubit Clifford generators" begin
        H = clifford_H()
        S = clifford_S()

        # H is Hermitian
        @test H ≈ H'

        # H is unitary
        @test H * H' ≈ Matrix{ComplexF64}(I, 2, 2)

        # S is unitary
        @test S * S' ≈ Matrix{ComplexF64}(I, 2, 2)

        # S^4 = I
        @test S^4 ≈ Matrix{ComplexF64}(I, 2, 2)
    end

    @testset "CNOT gate" begin
        CNOT = clifford_CNOT()
        I4 = Matrix{ComplexF64}(I, 4, 4)

        # CNOT is unitary
        @test CNOT * CNOT' ≈ I4

        # CNOT is self-inverse
        @test CNOT * CNOT ≈ I4
    end

    @testset "Clifford group enumeration" begin
        group = generate_clifford_group()

        # Exactly 720 elements
        @test length(group) == 720

        # All elements are unitary
        I4 = Matrix{ComplexF64}(I, 4, 4)
        for C in group
            @test C * C' ≈ I4 atol=1e-10
        end
    end

    @testset "Identity in group" begin
        group = generate_clifford_group()
        I4 = Matrix{ComplexF64}(I, 4, 4)

        # Identity is in the group (modulo phase)
        has_identity = any(C -> norm(C - I4) < 1e-10 ||
                                norm(C + I4) < 1e-10 ||
                                norm(C - im*I4) < 1e-10 ||
                                norm(C + im*I4) < 1e-10, group)
        @test has_identity
    end

    @testset "Clifford preserves Pauli group" begin
        group = generate_clifford_group()
        paulis = pauli_basis_matrices()

        # Test a sample of Cliffords (first 20 + last 10)
        sample_indices = vcat(1:min(20, length(group)),
                              max(1, length(group)-9):length(group))
        unique!(sample_indices)

        for idx in sample_indices
            C = group[idx]
            for P in paulis
                Q = C * P * C'
                # Q should be a Pauli (up to phase): Q = phase * P_j
                found = false
                for Pj in paulis
                    # Try to find phase such that Q = phase * Pj
                    # Find first nonzero entry ratio
                    r = zero(ComplexF64)
                    for i in 1:4, j in 1:4
                        if abs(Pj[i,j]) > 0.5
                            r = Q[i,j] / Pj[i,j]
                            break
                        end
                    end
                    if abs(r) > 0.5 && norm(Q - r * Pj) < 1e-10
                        # Phase should be in {1, -1, i, -i}
                        @test (abs(r - 1) < 1e-10 || abs(r + 1) < 1e-10 ||
                               abs(r - im) < 1e-10 || abs(r + im) < 1e-10)
                        found = true
                        break
                    end
                end
                @test found
            end
        end
    end

end

@testset "Pauli decomposition" begin
    # Identity decomposes to coeff 1 on I*I, 0 elsewhere
    I2 = ComplexF64[1 0; 0 1]
    @tensor h_identity[i,j,k,l] := I2[i,j] * I2[k,l]
    coeffs = pauli_decompose(h_identity)
    @test length(coeffs) == 16
    @test abs(coeffs[1] - 1.0) < 1e-12
    @test all(abs.(coeffs[2:end]) .< 1e-12)

    # Sx*Sx: coefficient at position 6 (index (X,X) = (2-1)*4+2 = 6)
    Sx = ComplexF64[0 1; 1 0]
    @tensor hxx[i,j,k,l] := Sx[i,j] * Sx[k,l]
    coeffs_xx = pauli_decompose(hxx)
    @test abs(coeffs_xx[6] - 1.0) < 1e-12
end

@testset "Pauli roundtrip" begin
    h = randn(ComplexF64, 2, 2, 2, 2)
    h_herm = zeros(ComplexF64, 2, 2, 2, 2)
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        h_herm[i,j,k,l] = (h[i,j,k,l] + conj(h[j,i,l,k])) / 2
    end
    coeffs = pauli_decompose(h_herm)
    h_recon = pauli_compose(coeffs)
    @test h_recon ≈ h_herm atol=1e-10
end

@testset "Clifford Hamiltonian transformation" begin
    # H*I transforms Sz*Sz -> Sx*Sz
    Sz = ComplexF64[1 0; 0 -1]
    @tensor hzz[i,j,k,l] := Sz[i,j] * Sz[k,l]
    H_gate = clifford_H()
    I2 = Matrix{ComplexF64}(I, 2, 2)
    C = kron(H_gate, I2)
    h_t = transform_bond_hamiltonian(hzz, C)
    Sx = ComplexF64[0 1; 1 0]
    @tensor hxz[i,j,k,l] := Sx[i,j] * Sz[k,l]
    @test h_t ≈ hxz atol=1e-10
end

@testset "Clifford preserves eigenvalues" begin
    Sx = const_Sx(0.5)
    Sy = const_Sy(0.5)
    Sz = const_Sz(0.5)
    @tensor h[i,j,k,l] := Sx[i,j]*Sx[k,l] + Sy[i,j]*Sy[k,l] + Sz[i,j]*Sz[k,l]
    group = generate_clifford_group()
    # Use kron-ordered matrix for eigenvalues (consistent with transform_bond_hamiltonian)
    h_mat = TeneT._tensor_to_mat(ComplexF64.(h))
    eig_orig = sort(real.(eigvals(h_mat)))
    for C in group[1:20]
        h_t = transform_bond_hamiltonian(ComplexF64.(h), C)
        h_t_mat = TeneT._tensor_to_mat(h_t)
        eig_t = sort(real.(eigvals(h_t_mat)))
        @test eig_orig ≈ eig_t atol=1e-10
    end
end

@testset "Bond entanglement entropy" begin

    @testset "Product state has zero entanglement" begin
        # h = sigma_z x I has product eigenstates
        Sz = ComplexF64[1 0; 0 -1]
        I2 = ComplexF64[1 0; 0 1]
        @tensor h[i,j,k,l] := Sz[i,j] * I2[k,l]
        @test bond_entanglement_entropy(h) ≈ 0.0 atol=1e-10
    end

    @testset "Bell state has maximal entanglement" begin
        # AFM Heisenberg: Sx*Sx + Sy*Sy + Sz*Sz, ground state is singlet with S = ln(2)
        Sx = const_Sx(0.5)
        Sy = const_Sy(0.5)
        Sz = const_Sz(0.5)
        @tensor h[i,j,k,l] := Sx[i,j]*Sx[k,l] + Sy[i,j]*Sy[k,l] + Sz[i,j]*Sz[k,l]
        S = bond_entanglement_entropy(h)
        @test S ≈ log(2) atol=1e-10
    end

    @testset "Entanglement is non-negative" begin
        group = generate_clifford_group()
        Sz = ComplexF64[1 0; 0 -1]
        @tensor hzz[i,j,k,l] := Sz[i,j] * Sz[k,l]
        for C in group[1:10]
            h_t = transform_bond_hamiltonian(hzz, C)
            S = bond_entanglement_entropy(h_t)
            @test S >= -1e-14
        end
    end

    @testset "Total bond entanglement" begin
        Sx = ComplexF64[0 1; 1 0]
        Sy = ComplexF64[0 -im; im 0]
        Sz = ComplexF64[1 0; 0 -1]
        @tensor hx[i,j,k,l] := Sx[i,j] * Sx[k,l]
        @tensor hy[i,j,k,l] := Sy[i,j] * Sy[k,l]
        @tensor hz[i,j,k,l] := Sz[i,j] * Sz[k,l]
        S_total = total_bond_entanglement((hx, hy, hz))
        @test S_total ≈ bond_entanglement_entropy(hx) + bond_entanglement_entropy(hy) + bond_entanglement_entropy(hz)
    end

end

@testset "Clifford optimizer" begin
    @testset "Kitaev model optimization runs" begin
        model = Kitaev(lattice=Honeycomb(:brickwall), S=0.5, Jx=-1.0, Jy=-1.0, Jz=-1.0)
        result = optimize_clifford(model; n_layers=1, max_sweeps=2, verbosity=0)
        @test haskey(result, :h_transformed)
        @test haskey(result, :circuit)
        @test haskey(result, :entanglement_history)
        h_t = result[:h_transformed]
        @test length(h_t) == 3
        for h in h_t
            @test size(h) == (2, 2, 2, 2)
        end
    end

    @testset "Optimization does not increase entanglement" begin
        model = Kitaev(lattice=Honeycomb(:brickwall), S=0.5, Jx=-1.0, Jy=-1.0, Jz=-1.0)
        result = optimize_clifford(model; n_layers=1, max_sweeps=5, verbosity=0)
        hist = result[:entanglement_history]
        for i in 2:length(hist)
            @test hist[i] <= hist[i-1] + 1e-10
        end
    end

    @testset "Eigenvalues preserved after optimization" begin
        model = Kitaev(lattice=Honeycomb(:brickwall), S=0.5, Jx=-1.0, Jy=-1.0, Jz=-1.0)
        h_orig = hamiltonian(model)
        result = optimize_clifford(model; n_layers=1, max_sweeps=3, verbosity=0)
        h_t = result[:h_transformed]
        for (ho, ht) in zip(h_orig, h_t)
            eig_o = sort(real.(eigvals(TeneT._tensor_to_mat(ComplexF64.(ho)))))
            eig_t = sort(real.(eigvals(TeneT._tensor_to_mat(ComplexF64.(ht)))))
            @test eig_o ≈ eig_t atol=1e-10
        end
    end
end
