# ─── Leg type aliases ─────────────────────────────────────────────────────────

const leg3 = Union{<:AbstractArray{T, 3}, Vector{<:AbstractArray{T, 3}}, StructArray{<:Vector{<:AbstractArray{T, 3}}}} where T
const leg4 = Union{<:AbstractArray{T, 4}, Vector{<:AbstractArray{T, 4}}, StructArray{<:Vector{<:AbstractArray{T, 4}}}} where T
const leg5 = Union{<:AbstractArray{T, 5}, Vector{<:AbstractArray{T, 5}}, StructArray{<:Vector{<:AbstractArray{T, 5}}}} where T
const leg8 = Union{<:AbstractArray{T, 8}, Vector{<:AbstractArray{T, 8}}, StructArray{<:Vector{<:AbstractArray{T, 8}}}} where T

# ─── Simple eigenvalue solver ────────────────────────────────────────────────
function simple_eig(f, v; power_iter)
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
    v1 = orth_for_ad(v1)
    # λ = 0.0 + 0.0im
    # if ifvalue
        # λ = dot(v, f(v))
    # end
    return [λ], [v1]
end

# ─── Checkpointing ──────────────────────────────────────────────────────────

# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, x...; kwargs...) = f(x...; kwargs...)
Zygote.@adjoint checkpoint(f, args...; kwargs...) = f(args...; kwargs...), ȳ -> Zygote._pullback((args...) -> f(args...; kwargs...), args...)[2](ȳ)

# ─── Checkpointing with host-memory offload ────────────────────────────────
# Like `checkpoint`, but after the forward pass the differentiable array
# arguments are transferred to host memory (`Array`). The device copies then
# become unreferenced and can be freed by GC (or explicit CUDA.reclaim).
# During backward, the inputs are moved back to the original array type and
# the forward is re-run under Zygote to produce the pullback.
#
# On CPU (Array) this is equivalent to `checkpoint` (the "transfer" is a copy
# and collapses to the identity); its value is on GPU runs where it trades
# device VRAM for host RAM + two one-way transfers.
#
# Caveat: only the explicit `args` are offloaded. Any large tensors captured
# inside `f` as a closure still live on the device. To get real VRAM savings
# the heavy tensors must be passed explicitly as `args`.
_offload_to_host(x::AbstractArray{<:Number}) = Array(x)
_offload_to_host(x::AbstractArray) = map(_offload_to_host, x)
_offload_to_host(x::Tuple) = map(_offload_to_host, x)
_offload_to_host(x) = x

# Atype detection: returns Array / CuArray / ROCArray, or `nothing` if no
# array-like could be found in `x`. Specialisations for StructArray and
# runtime structs live in boundary_algorithm/environment.jl (where those
# types are defined).
_atype_of(x::AbstractArray{<:Number}) = _arraytype(x)
_atype_of(x::AbstractArray) = isempty(x) ? nothing : _atype_of(first(x))
_atype_of(x::Tuple) = begin
    for a in x
        at = _atype_of(a)
        at === nothing || return at
    end
    return nothing
end
_atype_of(x) = nothing

function _detect_target_atype(args)
    for a in args
        at = _atype_of(a)
        at === nothing || return at
    end
    return Array
end

# Reconstruct an on-device copy from a CPU copy, given the target atype.
# The pullback closure captures only `atype` and `args_cpu`, never the
# original `args`, so the device originals become eligible for GC/free.
_to_atype(atype, x::Array{<:Number}) = atype(x)
_to_atype(atype, x::AbstractArray) = map(a -> _to_atype(atype, a), x)
_to_atype(atype, x::Tuple) = map(a -> _to_atype(atype, a), x)
_to_atype(_atype, x) = x

checkpoint_offload(f, x...; kwargs...) = f(x...; kwargs...)
Zygote.@adjoint function checkpoint_offload(f, args...; kwargs...)
    y = f(args...; kwargs...)
    atype = _detect_target_atype(args)
    args_cpu = map(_offload_to_host, args)
    # NOTE: the returned closure deliberately does NOT reference `args` so
    # that Julia will not capture it; only `args_cpu` + `atype` survive.
    return y, function(ȳ)
        args_dev = map(a -> _to_atype(atype, a), args_cpu)
        Zygote._pullback((aa...) -> f(aa...; kwargs...), args_dev...)[2](ȳ)
    end
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