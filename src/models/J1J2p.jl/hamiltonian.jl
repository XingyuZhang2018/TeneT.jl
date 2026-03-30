export J1J2

"""
    J1J2p{L<:AbstractLattice}

J1-J2p Heisenberg model with nearest-neighbor coupling `J1` and next-nearest-neighbor coupling `J2p` on a given lattice, but only on the one triangle not two on the Honeycomb lattice.
"""
@kwdef mutable struct J1J2p{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Honeycomb{:brickwall}()
    S::Real = 1/2
    J1::Real = 1.0
    J2p::Real = 0.5
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette
    bondratio::Real = 1.0 # only used when couplingtype is not :uniform
end

"""
    hamiltonian(model::J1J2p)

Hamiltonian of the J1-J2p Heisenberg model. The J1 term is the same as the Heisenberg model, while the J2 term is a next-nearest-neighbor coupling. When `ifrotate=true`, applies a sublattice rotation `U = 2Sx` so that the antiferromagnetic Néel order becomes a uniform state, enabling efficient optimization with a single-site unit cell.
"""
function hamiltonian(model::J1J2p)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    if model.ifrotate
        h = -(@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) -
             (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
             (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
        U = Sx * 2
        @tensor h[i,j,k,l] = h[i,j,c,d] * U[k,c] * conj(U[l,d])
    else
        h = (@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) +
            (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
            (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
    end
    return real(h)
end
