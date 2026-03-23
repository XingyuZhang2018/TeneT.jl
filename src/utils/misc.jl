# ─── Leg type aliases ─────────────────────────────────────────────────────────

const leg3 = Union{<:AbstractArray{T, 3}, Vector{<:AbstractArray{T, 3}}, StructArray{<:Vector{<:AbstractArray{T, 3}}}} where T
const leg4 = Union{<:AbstractArray{T, 4}, Vector{<:AbstractArray{T, 4}}, StructArray{<:Vector{<:AbstractArray{T, 4}}}} where T
const leg5 = Union{<:AbstractArray{T, 5}, Vector{<:AbstractArray{T, 5}}, StructArray{<:Vector{<:AbstractArray{T, 5}}}} where T
const leg8 = Union{<:AbstractArray{T, 8}, Vector{<:AbstractArray{T, 8}}, StructArray{<:Vector{<:AbstractArray{T, 8}}}} where T

# ─── Reshape helpers ──────────────────────────────────────────────────────────

function _to_front(t)
    χ = size(t)[end]
    return reshape(t, χ, Int(prod(size(t))/χ))
end

function _to_tail(t)
    χ = size(t, 1)
    return reshape(t, Int(prod(size(t))/χ), χ)
end

# ─── Permutation helpers ─────────────────────────────────────────────────────

permute_fronttail(t::leg3) = permutedims(t, (3,2,1))
permute_fronttail(t::leg4) = permutedims(t, (4,2,3,1))
permute_fronttail(t::InnerProductVec) = RealVec(permute_fronttail(t.vec))
permute_fronttail(t::AbstractZero) = t

# ─── Simple eigenvalue solver ────────────────────────────────────────────────

orth_for_ad(v) = v
function simple_eig(f, v; power_iter, ifvalue=false)
    # λ = 1.0 + 1.0im
    # Zygote.@ignore begin # this is not correct when VUMPS does not converge
        # for _ in 1:power_iter
        #     v = f(v)
        #     λ′ = norm(v)
        #     v /= λ′
        #     abs(λ′ - λ) < 1e-8 && break
        #     λ = λ′
        # end
    # end
    for _ in 1:power_iter-1
        v = f(v)
        v /= norm(v)
    end

    v1 = f(v)
    λ = dot(v, v1)
    v1 /= norm(v1)
    # v = orth_for_ad(v)
    # λ = 0.0 + 0.0im
    # if ifvalue
        # λ = dot(v, f(v))
    # end
    return [λ], [v1]
end

# ─── QR / LQ with positive diagonal ─────────────────────────────────────────
# (qrpos and lqpos are defined elsewhere; this file provides simple_eig which
#  is the forward pass used by the AD rule in autodiff/simple_eig_ad.jl)

# ─── Matrix canonical form ───────────────────────────────────────────────────

function mcform(M)
    aM = Array(M)
    x = ein"ijil->jl"(aM)
    _, vh = Zygote.@ignore eigen(x)
    aM = ein"aj,(ijkl,lb)->iakb"(inv(vh),aM,vh)
    y = ein"ijkj->ik"(aM)
    _, vv = Zygote.@ignore eigen(y)
    aM = ein"(ai,ijkl),kb->ajbl"(inv(vv),aM,vv)
    aM = typeof(M)(aM)
    return vh, vv, aM
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
