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
