export Kitaev

"""
    Kitaev{L<:AbstractLattice}

Kitaev model with couplings `Jx`, `Jy`, `Jz` on a given lattice.
"""
@kwdef mutable struct Kitaev{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Honeycomb{:brickwall}()
    S::Real = 1/2
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
    bondratio::Real = 1.0 # bondratio < 1.0 for plaquette >1.0 for dimer
end

"""
    hamiltonian(model::Kitaev)

Return the Kitaev Hamiltonian as a two-site operator.
"""
function hamiltonian(model::Kitaev)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    @tensor hx[i,j,k,l] := Sx[i,j] * Sx[k,l]
    @tensor hy[i,j,k,l] := Sy[i,j] * Sy[k,l]
    @tensor hz[i,j,k,l] := Sz[i,j] * Sz[k,l]
    return real.((hx, hy, hz))
end