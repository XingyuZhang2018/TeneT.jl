# Two-qubit Clifford group enumeration
#
# The two-qubit Clifford group (modulo global phase) has 720 elements.
# Generated from {H x I, I x H, S x I, I x S, CNOT} via BFS, identifying elements
# that have identical Pauli conjugation action (fingerprint).

"""
    clifford_H()

Return the single-qubit Hadamard gate as a 2x2 complex matrix.
"""
function clifford_H()
    return ComplexF64[1 1; 1 -1] / sqrt(2)
end

"""
    clifford_S()

Return the single-qubit phase gate S = diag(1, i) as a 2x2 complex matrix.
"""
function clifford_S()
    return ComplexF64[1 0; 0 im]
end

"""
    clifford_CNOT()

Return the two-qubit CNOT gate as a 4x4 complex matrix.
Control on qubit 1, target on qubit 2.
"""
function clifford_CNOT()
    return ComplexF64[1 0 0 0;
                      0 1 0 0;
                      0 0 0 1;
                      0 0 1 0]
end

"""
    pauli_basis_matrices()

Return the 16 two-qubit Pauli matrices {I,X,Y,Z} x {I,X,Y,Z} as a vector
of 4x4 complex matrices.
"""
function pauli_basis_matrices()
    I2 = ComplexF64[1 0; 0 1]
    X  = ComplexF64[0 1; 1 0]
    Y  = ComplexF64[0 -im; im 0]
    Z  = ComplexF64[1 0; 0 -1]
    single = [I2, X, Y, Z]
    paulis = Vector{Matrix{ComplexF64}}(undef, 16)
    idx = 1
    for a in single
        for b in single
            paulis[idx] = kron(a, b)
            idx += 1
        end
    end
    return paulis
end

"""
    generate_clifford_group()

Generate the two-qubit Clifford group modulo global phase.
Returns a Vector of exactly 720 unique 4x4 unitary matrices.

The group is generated from {H x I, I x H, S x I, I x S, CNOT} via BFS.
Two Cliffords are identified if they have the same Pauli conjugation action:
for every Pauli P, C*P*C' maps to the same Pauli P' (up to phase).
The fingerprint tracks only WHICH Pauli each Pauli maps to, not the phase.
"""
function generate_clifford_group()
    I2 = ComplexF64[1 0; 0 1]
    H = clifford_H()
    S = clifford_S()

    generators = Matrix{ComplexF64}[
        kron(H, I2),   # H x I
        kron(I2, H),   # I x H
        kron(S, I2),   # S x I
        kron(I2, S),   # I x S
        clifford_CNOT()
    ]

    paulis = pauli_basis_matrices()
    # Skip the identity Pauli (index 1) since C*I*C'=I always.
    # Use only the 15 non-identity Paulis for fingerprinting.
    nontrivial_paulis = paulis[2:16]

    # Fingerprint: for each non-identity Pauli P, compute C*P*C' and determine
    # which Pauli it maps to (up to phase). Uses trace inner product for robustness:
    # tr(Pj' * Q) / 4 gives the overlap coefficient, nonzero for exactly one Pj.
    function pauli_fingerprint(C::Matrix{ComplexF64})
        fp = Vector{Int}(undef, 15)
        for (k, P) in enumerate(nontrivial_paulis)
            Q = C * P * C'
            for (j, Pj) in enumerate(nontrivial_paulis)
                coeff = tr(Pj' * Q) / 4
                if abs(coeff) > 0.5
                    fp[k] = j
                    break
                end
            end
        end
        return fp
    end

    # BFS over the group
    seen_fingerprints = Set{Vector{Int}}()
    queue = Vector{Matrix{ComplexF64}}()
    group = Vector{Matrix{ComplexF64}}()

    # Seed with identity
    I4 = Matrix{ComplexF64}(I, 4, 4)
    fp_id = pauli_fingerprint(I4)
    push!(seen_fingerprints, fp_id)
    push!(queue, I4)
    push!(group, I4)

    head = 1
    while head <= length(queue)
        C = queue[head]
        head += 1
        for G in generators
            Cnew = G * C
            fp = pauli_fingerprint(Cnew)
            if fp ∉ seen_fingerprints
                push!(seen_fingerprints, fp)
                push!(queue, Cnew)
                push!(group, Cnew)
            end
        end
    end

    return group
end
