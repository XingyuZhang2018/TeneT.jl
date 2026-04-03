export bond_entanglement_entropy, total_bond_entanglement

"""
    bond_entanglement_entropy(h::AbstractArray{T,4}) where T

Compute the entanglement entropy of the ground state of a two-body Hamiltonian `h`.

Given a rank-4 tensor `h[i,j,k,l]` representing a two-qubit Hamiltonian,
find the ground state and compute its von Neumann entanglement entropy
across the bipartition into the two qubits.
"""
function bond_entanglement_entropy(h::AbstractArray{T,4}) where T
    d1, d2, d3, d4 = size(h)
    # h[i,j,k,l] = operator with i=row_site1, j=col_site1, k=row_site2, l=col_site2
    # Permute to h[i,k,j,l] (bra1, bra2, ket1, ket2) then reshape to matrix
    H_mat = reshape(permutedims(h, (1,3,2,4)), d1 * d3, d2 * d4)
    # Make Hermitian for numerical stability
    H_mat = (H_mat + H_mat') / 2

    # Diagonalize to find ground state
    vals, vecs = eigen(Hermitian(H_mat))
    # Ground state is eigenvector with lowest eigenvalue
    psi = vecs[:, 1]

    # Reshape to (d1, d3) matrix for Schmidt decomposition (site1 x site2)
    psi_mat = reshape(psi, d1, d3)

    # SVD to get Schmidt coefficients
    sv = svd(psi_mat)
    sigma = sv.S

    # Compute von Neumann entropy S = -sum(p_k * log(p_k)) where p_k = sigma_k^2
    S = zero(real(T))
    for s in sigma
        p = real(s)^2
        if p > 1e-15
            S -= p * log(p)
        end
    end
    return S
end

"""
    total_bond_entanglement(h_bonds)

Sum of `bond_entanglement_entropy` over all bond Hamiltonians in `h_bonds`.

`h_bonds` can be a tuple or vector of rank-4 tensors.
"""
function total_bond_entanglement(h_bonds)
    return sum(bond_entanglement_entropy(h) for h in h_bonds)
end
