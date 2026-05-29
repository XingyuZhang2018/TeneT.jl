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

function simple_eig(f, v; power_iter, checkpoint_every=5,
                    segment_checkpoint::CheckpointMethod=Plain(),
                    f_final=nothing, final_polish_steps=0)
    polish_active = f_final !== nothing && final_polish_steps > 0
    n_polish = polish_active ? min(final_polish_steps, power_iter) : 0
    n_pre = power_iter - n_polish    # total f-calls using `f` (pre-polish)

    if !polish_active
        # Original path — single f, all `power_iter` calls go to f
        n = power_iter - 1
        if n > 0 && checkpoint_every > 0 && checkpoint_every < n
            while n > 0
                seg = min(checkpoint_every, n)
                v = checkpoint(segment_checkpoint, _power_iter_segment, f, v, seg)
                n -= seg
            end
        else
            for _ in 1:n
                v = f(v)
                v /= norm(v)
            end
        end
        v1 = f(v)
    else
        # Pre-polish: n_pre normalizing iters using `f`
        # Polish:     (n_polish - 1) normalizing iters + 1 final iter, all using `f_final`
        if n_pre > 0
            np = n_pre
            if checkpoint_every > 0 && checkpoint_every < np
                while np > 0
                    seg = min(checkpoint_every, np)
                    v = checkpoint(segment_checkpoint, _power_iter_segment, f, v, seg)
                    np -= seg
                end
            else
                for _ in 1:np
                    v = f(v)
                    v /= norm(v)
                end
            end
        end
        for _ in 1:(n_polish - 1)
            v = f_final(v)
            v /= norm(v)
        end
        v1 = f_final(v)
    end

    λ = dot(v, v1)
    v1 /= norm(v1)
    v1 = orth_for_ad(v1)
    return [λ], [v1]
end

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

# ─── Truncated SVD wrapper ──────────────────────────────────────────────────

function _check_D_trunc(D_trunc::Integer, maxrank::Integer)
    1 <= D_trunc <= maxrank || throw(ArgumentError("D_trunc must be between 1 and $maxrank"))
    return D_trunc
end

function _truncate_svd(F::SVD, D_trunc::Integer)
    k = _check_D_trunc(D_trunc, length(F.S))
    return SVD(F.U[:, 1:k], F.S[1:k], F.Vt[1:k, :])
end

function _svdsolve_to_svd(vals, lvecs, rvecs, D_trunc::Integer)
    k = _check_D_trunc(D_trunc, length(vals))
    U = hcat(lvecs[1:k]...)
    V = hcat(rvecs[1:k]...)
    return SVD(U, vals[1:k], V')
end

function _linear_map_maxrank(f, x0::AbstractVector)
    m = length(x0)
    n = try
        if f isa Tuple && length(f) == 2
            length(f[2](zero(x0)))
        else
            length(f(zero(x0), Val(true)))
        end
    catch
        m
    end
    return min(m, n)
end

"""
    svd(A::AbstractMatrix, D_trunc; kwargs...)
    svd(A::AbstractMatrix; D_trunc, kwargs...)
    svd(f, fadj, x0::AbstractVector, D_trunc; kwargs...)
    svd((f, fadj), x0::AbstractVector, D_trunc; kwargs...)

Compute `LinearAlgebra.svd(A; kwargs...)` and truncate the returned factorization
to rank `D_trunc`. Without `D_trunc`, this delegates to the standard
`LinearAlgebra.svd` implementation.

For a matrix-free linear map, pass the forward action `f(x) = A * x`, the adjoint
action `fadj(y) = A' * y`, and an initial vector `x0` in the codomain of `f`.
"""
function LinearAlgebra.svd(A::AbstractMatrix; D_trunc=nothing, kwargs...)
    F = invoke(LinearAlgebra.svd, Tuple{AbstractVecOrMat{eltype(A)}}, A; kwargs...)
    return D_trunc === nothing ? F : _truncate_svd(F, D_trunc)
end

function LinearAlgebra.svd(A::AbstractMatrix, D_trunc::Integer; kwargs...)
    F = svd(A; kwargs...)
    return _truncate_svd(F, D_trunc)
end

function LinearAlgebra.svd(f, x0::AbstractVector, D_trunc::Integer;
                           which=:LR,
                           warn_not_converged::Bool=true,
                           kwargs...)
    _check_D_trunc(D_trunc, _linear_map_maxrank(f, x0))
    vals, lvecs, rvecs, info = svdsolve(f, x0, D_trunc, which; kwargs...)
    warn_not_converged && info.converged < D_trunc &&
        @warn "svdsolve converged $(info.converged) / $D_trunc singular values" normres=info.normres
    return _svdsolve_to_svd(vals, lvecs, rvecs, D_trunc)
end

function LinearAlgebra.svd(f, m::Integer, D_trunc::Integer;
                           T::Type=Float64,
                           which=:LR,
                           warn_not_converged::Bool=true,
                           kwargs...)
    return svd(f, rand(T, m), D_trunc; which, warn_not_converged, kwargs...)
end

LinearAlgebra.svd(f, fadj, x0::AbstractVector, D_trunc::Integer; kwargs...) =
    svd((f, fadj), x0, D_trunc; kwargs...)

LinearAlgebra.svd(f, fadj, m::Integer, D_trunc::Integer; kwargs...) =
    svd((f, fadj), m, D_trunc; kwargs...)

# ─── Randomized SVD ─────────────────────────────────────────────────────────

_rsvd_random_eltype(::Type{T}) where {T<:AbstractFloat} = T
_rsvd_random_eltype(::Type{Complex{T}}) where {T<:AbstractFloat} = Complex{T}
_rsvd_random_eltype(::Type{T}) where {T<:Real} = Float64
_rsvd_random_eltype(::Type{Complex{T}}) where {T<:Real} = ComplexF64

_rsvd_orth(A::AbstractMatrix) = _mattype(A)(qr(A).Q)

function _rsvd_probe(n::Integer, blockdim::Integer, randtype::Type;
                     atype=Array,
                     rng=Random.default_rng(),
                     Omega=nothing)
    if Omega === nothing
        return atype(randn(rng, randtype, n, blockdim))
    end
    size(Omega) == (n, blockdim) ||
        throw(DimensionMismatch("Omega must have size ($n, $blockdim), got $(size(Omega))"))
    return Omega
end

function _check_rsvd_block(Y::AbstractMatrix, nrows::Integer, ncols::Integer, name::String)
    size(Y) == (nrows, ncols) ||
        throw(DimensionMismatch("$name returned size $(size(Y)), expected ($nrows, $ncols)"))
    return Y
end

"""
    rsvd(A, D_trunc; oversampling=10, niter=2, rng=Random.default_rng())
    rsvd(A; D_trunc, oversampling=10, niter=2, rng=Random.default_rng())
    rsvd(f, fadj, m, n, D_trunc; oversampling=10, niter=2, atype=Array, T=Float64, Omega=nothing)
    rsvd((f, fadj), m, n, D_trunc; kwargs...)

Compute a rank-`D_trunc` randomized singular value decomposition of matrix `A`.
Returns an `SVD` factorization like `LinearAlgebra.svd`, with
`F.U * Diagonal(F.S) * F.V' ≈ A`.

`oversampling` adds extra random probe vectors before truncation, while `niter`
sets the number of power iterations used to improve the singular subspace.
For matrix-free use, `f` and `fadj` should support block matrix inputs.
"""
function rsvd(A::AbstractMatrix, D_trunc::Integer;
              oversampling::Integer=10,
              niter::Integer=2,
              rng=Random.default_rng())
    m, n = size(A)
    maxrank = min(m, n)
    D_trunc = _check_D_trunc(D_trunc, maxrank)
    oversampling >= 0 || throw(ArgumentError("oversampling must be non-negative"))
    niter >= 0 || throw(ArgumentError("niter must be non-negative"))

    blockdim = min(n, D_trunc + oversampling)
    randtype = _rsvd_random_eltype(eltype(A))
    Ω = _rsvd_probe(n, blockdim, randtype; atype=_arraytype(A), rng)

    Q = _rsvd_orth(A * Ω)
    for _ in 1:niter
        Q = _rsvd_orth(A' * Q)
        Q = _rsvd_orth(A * Q)
    end

    F = svd(Q' * A)
    U = Q * F.U[:, 1:D_trunc]
    S = F.S[1:D_trunc]
    V = F.V[:, 1:D_trunc]
    return SVD(U, S, V')
end

rsvd(A::AbstractMatrix; D_trunc::Integer, kwargs...) = rsvd(A, D_trunc; kwargs...)

function rsvd(f, fadj, m::Integer, n::Integer, D_trunc::Integer; kwargs...)
    return rsvd((f, fadj), m, n, D_trunc; kwargs...)
end

function rsvd(fpair::Tuple, m::Integer, n::Integer, D_trunc::Integer;
              oversampling::Integer=10,
              niter::Integer=2,
              rng=Random.default_rng(),
              atype=Array,
              T::Type=Float64,
              Omega=nothing)
    length(fpair) == 2 || throw(ArgumentError("fpair must be a tuple (f, fadj)"))
    f, fadj = fpair
    maxrank = min(m, n)
    D_trunc = _check_D_trunc(D_trunc, maxrank)
    oversampling >= 0 || throw(ArgumentError("oversampling must be non-negative"))
    niter >= 0 || throw(ArgumentError("niter must be non-negative"))

    blockdim = min(n, D_trunc + oversampling)
    randtype = _rsvd_random_eltype(T)
    Ω = _rsvd_probe(n, blockdim, randtype; atype, rng, Omega)

    Y = _check_rsvd_block(f(Ω), m, blockdim, "f(Omega)")
    Q = _rsvd_orth(Y)
    for _ in 1:niter
        Z = _check_rsvd_block(fadj(Q), n, size(Q, 2), "fadj(Q)")
        Q = _rsvd_orth(_check_rsvd_block(f(Z), m, size(Z, 2), "f(fadj(Q))"))
    end

    Z = _check_rsvd_block(fadj(Q), n, size(Q, 2), "fadj(Q)")
    F = svd(Z')
    U = Q * F.U[:, 1:D_trunc]
    S = F.S[1:D_trunc]
    V = F.V[:, 1:D_trunc]
    return SVD(U, S, V')
end

rsvd(f, m::Integer, n::Integer, D_trunc::Integer; kwargs...) =
    rsvd((X -> f(X, Val(false)), X -> f(X, Val(true))), m, n, D_trunc; kwargs...)
