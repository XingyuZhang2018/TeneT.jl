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
    @tensor B9[l, lb, d, db, r, rb, u, ub, zout] := A[l, d, r, u, zin] * W[lb, db, rb, ub, zin, zout]
    s = size(A)
    return reshape(B9, 2s[1], 2s[2], 2s[3], 2s[4], 2)
end

function apply_transfer(model::CubicDimer, A::StructArray)
    return StructArray([apply_transfer(model, a) for a in A.data], A.pattern)
end

"""
    transfer_layer(model::CubicDimer, A::StructArray)

Build the mixed triple-layer object `(T*A, conj(A))` consumed by the existing
tuple `FLmap`/`FRmap`/`ACmap` kernels.
"""
function transfer_layer(model::CubicDimer, A::StructArray)
    B = apply_transfer(model, A)
    return StructArray([(B.data[i], conj(A.data[i])) for i in eachindex(A.data)], A.pattern)
end

mixed_transfer_tensor(model::CubicDimer, A::StructArray) = transfer_layer(model, A)
