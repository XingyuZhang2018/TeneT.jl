# Shastry-Sutherland model

"""
    SS{L}(; lattice, S, J1, J2)

Shastry-Sutherland model on a given lattice.
"""
@kwdef struct SS{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.0
end

function hamiltonian(model::SS)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = (@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) +
        (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
        (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
    return h
end
