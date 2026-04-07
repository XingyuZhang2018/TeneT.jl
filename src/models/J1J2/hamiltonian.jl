export J1J2

"""
    J1J2{L<:AbstractLattice}

J1-J2 Heisenberg model with nearest-neighbor coupling `J1` and next-nearest-neighbor coupling `J2` on a given lattice.
"""
@kwdef mutable struct J1J2{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.5
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette, :dimmer1, :dimmer2, :mixed
    bondratio::Real = 1.0 # only used when couplingtype is not :uniform
                          # for Honeycomb{:brickwall}, bondratio<1 is plaquette, bondratio>1 is dimmer
end

"""
    hamiltonian(model::J1J2)

Hamiltonian of the J1-J2 Heisenberg model. The J1 term is the same as the Heisenberg model, while the J2 term is a next-nearest-neighbor coupling. When `ifrotate=true`, applies a sublattice rotation `U = 2Sx` so that the antiferromagnetic Néel order becomes a uniform state, enabling efficient optimization with a single-site unit cell.
"""
function hamiltonian(model::J1J2)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = (@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) +
        (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
        (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
    if model.ifrotate
        U = Sy * 2
        @tensor h_rot[i,j,k,l] := h[i,j,c,d] * U[k,c] * conj(U[l,d])
        h = h_rot
    end
    return real(h)
end
