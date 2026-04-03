export TransformedKitaev

struct TransformedKitaev{L<:AbstractLattice} <: HamiltonianModel
    original::Kitaev{L}
    h_transformed::NTuple{3, Array{Float64, 4}}
    clifford_circuit::Vector
end

# hamiltonian() returns the transformed operators
function hamiltonian(model::TransformedKitaev{Honeycomb{:brickwall}})
    return model.h_transformed
end

# Forward lattice/coupling accessors to the original model
Base.getproperty(m::TransformedKitaev, s::Symbol) =
    s in (:original, :h_transformed, :clifford_circuit) ? getfield(m, s) :
    getproperty(getfield(m, :original), s)

# Forward enlarge_coupling to original
function enlarge_coupling(model::TransformedKitaev{Honeycomb{:brickwall}}, i::Int, j::Int)
    return enlarge_coupling(getfield(model, :original), i, j)
end

function enlarge_coupling(model::TransformedKitaev{Honeycomb{:brickwall}}, ::Val{V}, i, j) where V
    return enlarge_coupling(getfield(model, :original), Val(V), i, j)
end

# Energy dispatch delegates to shared helper
function energy_value(model::TransformedKitaev{Honeycomb{:brickwall}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    return _kitaev_brickwall_energy(model, A, env, params)
end

# Show
function Base.show(io::IO, m::TransformedKitaev)
    print(io, "TransformedKitaev(", getfield(m, :original), ", circuit_depth=", length(getfield(m, :clifford_circuit)), ")")
end
