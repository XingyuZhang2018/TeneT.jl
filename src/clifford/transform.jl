# Pauli decomposition and Clifford Hamiltonian transformation
#
# Tools for decomposing two-qubit bond Hamiltonians into the Pauli basis
# and applying Clifford transformations.
#
# Convention: bond Hamiltonians are rank-4 tensors h[i,j,k,l] where
# (i,j) are the row/column of qubit 1 and (k,l) are row/column of qubit 2.
# The corresponding 4x4 matrix uses the kron ordering:
#   h_mat[(i-1)*d+k, (j-1)*d+l] = h[i,j,k,l]
# which matches kron(A,B) when h[i,j,k,l] = A[i,j]*B[k,l].

export pauli_decompose, pauli_compose, transform_bond_hamiltonian

"""
    _tensor_to_mat(h::AbstractArray{T,4}) where T

Convert a (d,d,d,d) bond Hamiltonian tensor to the (d^2,d^2) matrix
consistent with the kron convention used by `pauli_basis_matrices()`.

Maps h[i,j,k,l] -> mat[(i-1)*d+k, (j-1)*d+l].
"""
function _tensor_to_mat(h::AbstractArray{T,4}) where T
    d = size(h, 1)
    mat = zeros(T, d*d, d*d)
    for i in 1:d, j in 1:d, k in 1:d, l in 1:d
        mat[(i-1)*d+k, (j-1)*d+l] = h[i,j,k,l]
    end
    return mat
end

"""
    _mat_to_tensor(h_mat::AbstractMatrix{T}) where T

Convert a (d^2,d^2) matrix back to a (d,d,d,d) tensor,
inverse of `_tensor_to_mat`.
"""
function _mat_to_tensor(h_mat::AbstractMatrix{T}) where T
    d = isqrt(size(h_mat, 1))
    h = zeros(T, d, d, d, d)
    for i in 1:d, j in 1:d, k in 1:d, l in 1:d
        h[i,j,k,l] = h_mat[(i-1)*d+k, (j-1)*d+l]
    end
    return h
end

"""
    pauli_decompose(h::AbstractArray{T,4}) where T

Decompose a (d,d,d,d) tensor (d=2) into 16 Pauli basis coefficients.

The coefficient for Pauli basis element P[alpha] is:
    c[alpha] = (1/4) tr(P[alpha]' * h_mat)
where h_mat is the 4x4 matrix in the kron convention.

Returns a length-16 vector of coefficients in the order
{I,X,Y,Z} x {I,X,Y,Z}.
"""
function pauli_decompose(h::AbstractArray{T,4}) where T
    paulis = pauli_basis_matrices()
    h_mat = _tensor_to_mat(h)
    coeffs = Vector{ComplexF64}(undef, 16)
    for (i, P) in enumerate(paulis)
        coeffs[i] = tr(P' * h_mat) / 4
    end
    return coeffs
end

"""
    pauli_compose(coeffs::AbstractVector)

Reconstruct a (2,2,2,2) tensor from 16 Pauli basis coefficients.

Computes h_mat = sum(c * P for (c, P) in zip(coeffs, paulis))
and returns the result as a (2,2,2,2) tensor.
"""
function pauli_compose(coeffs::AbstractVector)
    paulis = pauli_basis_matrices()
    h_mat = sum(c * P for (c, P) in zip(coeffs, paulis))
    return _mat_to_tensor(h_mat)
end

"""
    transform_bond_hamiltonian(h::AbstractArray{T,4}, C::AbstractMatrix) where T

Apply a Clifford transformation to a bond Hamiltonian: h -> C h C'.

h is a (d,d,d,d) tensor, C is a (d^2, d^2) unitary matrix.
Returns the transformed tensor as (d,d,d,d).
"""
function transform_bond_hamiltonian(h::AbstractArray{T,4}, C::AbstractMatrix) where T
    h_mat = _tensor_to_mat(h)
    h_new = C * h_mat * C'
    return _mat_to_tensor(h_new)
end
