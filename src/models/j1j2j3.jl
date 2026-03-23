"""
    J1J2J3{L}(; lattice, S, J1, J2, J3)

J1-J2-J3 model on a given lattice.
"""
@kwdef struct J1J2J3{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 1.0
    J3::Real = 1.0
end

function hamiltonian(model::J1J2J3)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = (@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) +
        (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
        (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
    return real(h)
end
