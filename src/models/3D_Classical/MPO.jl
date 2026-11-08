export CubicDimer, transfer_pepo, dimer_pepo, apply_transfer
export transfer_layer, mixed_transfer_tensor

"""
    CubicDimer(; lattice=Square())

Classical close-packed dimer covering model on the cubic lattice, viewed as a
2D transfer problem along the third lattice direction. The PEPS physical leg
stores the vertical dimer occupation (`0/1`), so `S=1/2` gives the required
two-state local space when using `init_ipeps`.
"""
@kwdef mutable struct CubicDimer{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
end

function _check_cubic_dimer_square(model::CubicDimer)
    model.lattice isa Square ||
        throw(ArgumentError("CubicDimer currently supports the square transfer layer; got $(typeof(model.lattice))"))
    model.S == 1/2 ||
        throw(ArgumentError("CubicDimer uses a two-state vertical occupation leg; expected S=1/2, got $(model.S)"))
    return nothing
end

"""
    dimer_pepo(model::CubicDimer; atype=Array, etype=Float64)

Return the local cubic-lattice dimer transfer tensor
`T[l,d,r,u,zin,zout]`, with binary indices encoded as `1 => 0` and
`2 => 1`. A tensor element is one iff exactly one of the six incident bonds is
occupied.
"""
function dimer_pepo(model::CubicDimer; atype=Array, etype=Float64)
    _check_cubic_dimer_square(model)
    T = zeros(etype, 2, 2, 2, 2, 2, 2)
    @inbounds for l in 1:2, d in 1:2, r in 1:2, u in 1:2, zin in 1:2, zout in 1:2
        T[l, d, r, u, zin, zout] = (l + d + r + u + zin + zout == 7) ? one(etype) : zero(etype)
    end
    return atype(T)
end

transfer_pepo(model::CubicDimer; kwargs...) = dimer_pepo(model; kwargs...)

_dimer_weight_eltype(::Type{T}) where {T} = T <: Complex ? real(T) : T

"""
    LayeredPEPOSite(W, A, Ac)

Mixed transfer site for a PEPO-applied ket tensor `A` and bra tensor `Ac`.
The enlarged ket tensor `WA = W*A` is cached so the contraction maps can use
the optimized leg5-pair kernels while keeping the PEPO ingredients available
for diagnostics and future model-specific observables.
"""
struct LayeredPEPOSite{WT<:AbstractArray,AT<:AbstractArray,BT<:AbstractArray,CT<:AbstractArray}
    W::WT
    A::AT
    Ac::BT
    WA::CT
end

function LayeredPEPOSite(W::AbstractArray, A::AbstractArray{<:Number,5},
                         Ac::AbstractArray{<:Number,5}, WA::AbstractArray{<:Number,5})
    size(W, 5) == size(A, 5) ||
        throw(ArgumentError("PEPO input physical dimension $(size(W, 5)) does not match A dimension $(size(A, 5))"))
    size(W, 6) == size(Ac, 5) ||
        throw(ArgumentError("PEPO output physical dimension $(size(W, 6)) does not match Ac dimension $(size(Ac, 5))"))
    size(WA, 5) == size(Ac, 5) ||
        throw(ArgumentError("cached PEPO-applied tensor physical dimension $(size(WA, 5)) does not match Ac dimension $(size(Ac, 5))"))
    return LayeredPEPOSite{typeof(W),typeof(A),typeof(Ac),typeof(WA)}(W, A, Ac, WA)
end

Base.eltype(M::LayeredPEPOSite) = promote_type(eltype(M.W), eltype(M.A), eltype(M.Ac), eltype(M.WA))
_arraytype(M::LayeredPEPOSite) = _arraytype(M.WA)
get_device_id(M::LayeredPEPOSite) = get_device_id(M.WA)

_downcast_eltype(::Nothing, M::LayeredPEPOSite) = M
function _downcast_eltype(T::Type, M::LayeredPEPOSite)
    return LayeredPEPOSite(_downcast_eltype(T, M.W),
                           _downcast_eltype(T, M.A),
                           _downcast_eltype(T, M.Ac),
                           _downcast_eltype(T, M.WA))
end
_boundary_cast(T::Type, M::LayeredPEPOSite) = _downcast_eltype(T, M)

_layered_fused_dim(M::LayeredPEPOSite, leg::Integer) = size(M.WA, leg)

"""
    apply_transfer(model::CubicDimer, A)

Apply the local cubic-dimer PEPO to an iPEPS site tensor
`A[left, down, right, up, zin]`. Each virtual leg is fused with the
corresponding binary dimer PEPO bond, returning
`B[(left,lbit), (down,dbit), (right,rbit), (up,ubit), zout]`.
"""
function apply_transfer(model::CubicDimer, A::AbstractArray{T,5}) where {T}
    _check_cubic_dimer_square(model)
    size(A, 5) == 2 ||
        throw(ArgumentError("CubicDimer expects physical dimension 2; got $(size(A, 5))"))
    W = Zygote.@ignore dimer_pepo(model; atype=_arraytype(A), etype=_dimer_weight_eltype(T))
    return _apply_transfer(W, A)
end

function _apply_transfer(W::AbstractArray, A::AbstractArray{<:Number,5})
    @tensor B9[l, lb, d, db, r, rb, u, ub, zout] := A[l, d, r, u, zin] * W[lb, db, rb, ub, zin, zout]
    s = size(A)
    return reshape(B9, 2s[1], 2s[2], 2s[3], 2s[4], 2)
end

function apply_transfer(model::CubicDimer, A::StructArray)
    return StructArray([apply_transfer(model, a) for a in A.data], A.pattern)
end

"""
    transfer_layer(model::CubicDimer, A::StructArray)

Build a mixed transfer layer for the transfer-PEPO optimization. The returned
sites cache the PEPO-applied ket tensor `W*A` and its bra partner `conj(A)`.
"""
function transfer_layer(model::CubicDimer, A::StructArray)
    _check_cubic_dimer_square(model)
    W = Zygote.@ignore dimer_pepo(
        model;
        atype=_arraytype(A),
        etype=_dimer_weight_eltype(eltype(A.data[1])),
    )
    return StructArray([
        LayeredPEPOSite(W, a, conj(a), _apply_transfer(W, a))
        for a in A.data
    ], A.pattern)
end

"""
    mixed_transfer_tensor(model::CubicDimer, A::StructArray)

Materialize the legacy mixed tuple `(T*A, conj(A))`. This is kept for
compatibility and for tests that compare the lazy layered kernels against the
original contraction path.
"""
function mixed_transfer_tensor(model::CubicDimer, A::StructArray)
    B = apply_transfer(model, A)
    return StructArray([(B.data[i], conj(A.data[i])) for i in eachindex(A.data)], A.pattern)
end

FLmap(FL, ALu, ALd, M::LayeredPEPOSite; inner_etype=nothing) =
    FLmap(FL, ALu, ALd, M.WA, M.Ac; inner_etype)

FRmap(FR, ARu, ARd, M::LayeredPEPOSite; inner_etype=nothing) =
    FRmap(FR, ARu, ARd, M.WA, M.Ac; inner_etype)

ACmap(AC, FL, FR, M::LayeredPEPOSite; inner_etype=nothing) =
    ACmap(AC, FL, FR, M.WA, M.Ac; inner_etype)

function FLmap_parallel(FL, ALu, ALd, M::LayeredPEPOSite;
                        ifparallel, forloop_iter, inner_etype=nothing,
                        comm=MPI.COMM_WORLD)
    N_in = (3, ndims(ALd))
    N_out = ndims(ALd)
    size_out = (size(FL, 1), _layered_fused_dim(M, 3), size(M.Ac, 3), size(FL, 4))
    if ifparallel
        return parallel(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out, inner_etype, comm)
    else
        return forloop(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function FRmap_parallel(FR, ARu, ARd, M::LayeredPEPOSite;
                        ifparallel, forloop_iter, inner_etype=nothing,
                        comm=MPI.COMM_WORLD)
    N_in = (3, 1)
    N_out = ndims(ARd)
    size_out = (size(ARd, 1), _layered_fused_dim(M, 1), size(M.Ac, 1), size(ARd, ndims(ARd)))
    if ifparallel
        return parallel(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out, inner_etype, comm)
    else
        return forloop(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function ACmap_parallel(AC, FL, FR, M::LayeredPEPOSite;
                        ifparallel, forloop_iter, inner_etype=nothing,
                        comm=MPI.COMM_WORLD)
    N_in = (3, ndims(FR))
    N_out = ndims(FR)
    size_out = (size(FR, 1), _layered_fused_dim(M, 2), size(M.Ac, 2), size(FR, ndims(FR)))
    if ifparallel
        return parallel(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype, comm)
    else
        return forloop(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function _randSA_for_layered_sites(M::StructArray, sizes)
    m = M.data[1]
    return randSA(eltype(m), _arraytype(m), M.pattern, sizes)
end

function FLint(AL, M::StructArray{<:Vector{<:LayeredPEPOSite}})
    chi = size(AL[1], 1)
    return _randSA_for_layered_sites(M, [(chi, _layered_fused_dim(m, 1), size(m.Ac, 1), chi) for m in M.data])
end

function FRint(AR, M::StructArray{<:Vector{<:LayeredPEPOSite}})
    chi = size(AR[1], 1)
    return _randSA_for_layered_sites(M, [(chi, _layered_fused_dim(m, 3), size(m.Ac, 3), chi) for m in M.data])
end

function initial_A(M::StructArray{<:Vector{<:LayeredPEPOSite}}, chi::Int)
    return _randSA_for_layered_sites(M, [(chi, _layered_fused_dim(m, 4), size(m.Ac, 4), chi) for m in M.data])
end

function _down_m(M::LayeredPEPOSite)
    return LayeredPEPOSite(permutedims(M.W, (1, 4, 3, 2, 5, 6)),
                           permutedims(M.A, (1, 4, 3, 2, 5)),
                           permutedims(M.Ac, (1, 4, 3, 2, 5)),
                           permutedims(M.WA, (1, 4, 3, 2, 5)))
end

_c4v_local_tensor(M::LayeredPEPOSite) = M

function _c4v_initial_FL(M::LayeredPEPOSite, chi::Int)
    FL = rand!(similar(M.A, eltype(M), chi, _layered_fused_dim(M, 1), size(M.Ac, 1), chi))
    FL += conj(permutedims(FL, (4, 2, 3, 1)))
    return FL
end
