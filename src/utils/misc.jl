# ─── Leg type aliases ─────────────────────────────────────────────────────────

const leg3 = Union{<:AbstractArray{T, 3}, Vector{<:AbstractArray{T, 3}}, StructArray{<:Vector{<:AbstractArray{T, 3}}}} where T
const leg4 = Union{<:AbstractArray{T, 4}, Vector{<:AbstractArray{T, 4}}, StructArray{<:Vector{<:AbstractArray{T, 4}}}} where T
const leg5 = Union{<:AbstractArray{T, 5}, Vector{<:AbstractArray{T, 5}}, StructArray{<:Vector{<:AbstractArray{T, 5}}}} where T
const leg8 = Union{<:AbstractArray{T, 8}, Vector{<:AbstractArray{T, 8}}, StructArray{<:Vector{<:AbstractArray{T, 8}}}} where T

# ─── Simple eigenvalue solver ────────────────────────────────────────────────
"""
    _power_iter_segment(f, v, n)

Run `n` steps of power iteration: v = f(v) / norm(v).
Used as a checkpoint-able segment inside simple_eig.
"""
function _power_iter_segment(f, v, n)
    for _ in 1:n
        v = f(v)
        v /= norm(v)
    end
    return v
end

function simple_eig(f, v; power_iter, checkpoint_every=5)
    n = power_iter - 1
    if n > 0 && checkpoint_every > 0 && checkpoint_every < n
        # Split into segments, each checkpointed
        while n > 0
            seg = min(checkpoint_every, n)
            v = checkpoint(_power_iter_segment, f, v, seg)
            n -= seg
        end
    else
        for _ in 1:n
            v = f(v)
            v /= norm(v)
        end
    end

    v1 = f(v)
    λ = dot(v, v1)
    v1 /= norm(v1)
    v1 = orth_for_ad(v1)
    return [λ], [v1]
end

# ─── Checkpointing ──────────────────────────────────────────────────────────

# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, x...; kwargs...) = f(x...; kwargs...)
Zygote.@adjoint checkpoint(f, args...; kwargs...) = f(args...; kwargs...), ȳ -> Zygote._pullback((args...) -> f(args...; kwargs...), args...)[2](ȳ)

# ─── Takagi decomposition ───────────────────────────────────────────────────

"""
    takagi_decomposition(M; D_trunc)

Perform Takagi factorization for complex symmetric matrices.
Decomposes matrix `M` into `M = A * transpose(A)` where:
- `M` is a complex symmetric matrix (M = M^T)
- `D_trunc`: truncation dimension to retain in the decomposition

This implementation:
1. Diagonalizes M†M to find eigenvectors (V) and eigenvalues (λ)
2. Extracts singular values from diagonal matrix D = VᵀMV
3. Constructs matrix A using first D_trunc eigenvectors scaled by sqrt(singular values)

Primarily used for SU parameterization in iPEPS tensor network simulations.
"""
function takagi_decomposition(M; D_trunc)
    norm(M - transpose(M)) < 1e-10 || throw(ArgumentError("M should be a complex symmetric matrix"))
    # Diagonalize M†M and sort eigenvectors by descending eigenvalues
    _, V = eigen(M'*M; sortby=x->-x)

    # Calculate diagonal matrix of singular values squared (D = VᵀMV)
    D = diag(transpose(V) * M * V)

    # Construct decomposition matrix: A = V*_trunc * sqrt(D_trunc)
    A = conj(V[:,1:D_trunc]) * diagm(sqrt.(D[1:D_trunc]))

    return A
end

# ─── Positive QR decomposition ───────────────────────────────────────────────────

safesign(x::Number) = iszero(x) ? one(x) : sign(x)

"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A))
    Q = mattype(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    Q .= Q * Diagonal(phases)
    R .= Diagonal(conj.(phases)) * R
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A'))
    Q = mattype(mattype(F.Q)')
    L = mattype(F.R')
    phases = safesign.(diag(L))
    Q .= Diagonal(phases) * Q
    L .= L * Diagonal(conj!(phases))
    return L, Q
end

"""
    qr_for_ad(A)

AD-friendly QR decomposition that returns a concrete Q matrix (not a `QRCompactWYQ`).
Unlike `qrpos`, does NOT enforce positive diagonal on R, which is important for
complex-valued tensors and AD stability.
"""
function qr_for_ad(A::AbstractMatrix{T}) where {T}
    Q, R = qr(A)
    Q = _arraytype(A)(Q)
    return Q, R
end