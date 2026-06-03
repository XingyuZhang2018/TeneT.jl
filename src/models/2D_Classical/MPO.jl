export Ising, MPO

"""
    Ising{L}(; lattice=Square(), beta=0.4)

Two-dimensional classical Ising model on a lattice at inverse temperature
`beta`. The current tensor builders support the square lattice.
"""
@kwdef mutable struct Ising{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    beta::Real = 0.4
end

function MPO(model::HamiltonianModel, args...; kwargs...)
    throw(ArgumentError("MPO is not implemented for $(typeof(model))"))
end

function _ising_weight_sqrt(beta::Real)
    B = [exp(beta) exp(-beta);
         exp(-beta) exp(beta)]
    return sqrt(B)
end

"""
    MPO(model::Ising{Square}; atype=Array, pattern=[1;;])

Build the corrected square-lattice 2D classical Ising tensor as a rank-4
`StructArray`. The local tensor is
`M[a,b,c,d] = sum_s W[s,a] W[s,b] W[s,c] W[s,d]`, where `W = sqrt(B)`.
"""
function MPO(model::Ising{Square}; atype=Array, pattern=[1;;])
    W = _ising_weight_sqrt(model.beta)
    d = size(W, 2)
    M = zeros(eltype(W), d, d, d, d)

    @inbounds for a in 1:d, b in 1:d, c in 1:d, e in 1:d, s in 1:d
        M[a, b, c, e] += W[s, a] * W[s, b] * W[s, c] * W[s, e]
    end

    return StructArray([atype(M)], pattern)
end

MPO(model::Ising{Square}, ::Type{General}; kwargs...) = MPO(model; kwargs...)
MPO(model::Ising{Square}, ::General; kwargs...) = MPO(model; kwargs...)

function MPO(model::Ising; kwargs...)
    throw(ArgumentError("MPO currently supports Ising{Square}; got $(typeof(model.lattice))"))
end

"""
    MPO(model::Ising{Square}, C4v; atype=Array)

Build the rank-5 C4v transfer tensor expected by `VUMPS{C4v}` by adding a
trivial physical leg to the rank-4 square-lattice Ising tensor.
"""
function MPO(model::Ising{Square}, ::Type{C4v}; atype=Array)
    M4 = MPO(model; atype).data[1]
    D = size(M4, 1)
    M5 = reshape(M4, D, D, D, D, 1)
    return StructArray([M5], [1;;])
end

MPO(model::Ising{Square}, ::C4v; kwargs...) = MPO(model, C4v; kwargs...)
