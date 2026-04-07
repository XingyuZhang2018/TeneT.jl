export J1J2J3

"""
    J1J2J3{L<:AbstractLattice}

J1-J2-J3 Heisenberg model with nearest-neighbor coupling `J1` and next-nearest-neighbor coupling `J2` and next-next-nearest-neighbor coupling `J3`. The model is defined on a lattice `L`, which can be specified by the user (e.g., `Square()`, `Honeycomb(:brickwall)`, etc.). The spin magnitude is given by `S`. The parameter `ifrotate` determines whether to rotate the spin operators in the Hamiltonian, and `couplingtype` specifies the type of coupling pattern (e.g., uniform, plaquette, dimmer, etc.). The `bondratio` parameter is used to adjust the coupling strength for non-uniform patterns.
"""
@kwdef mutable struct J1J2J3{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.6
    J3::Real = 0.4
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette, :dimmer1, :dimmer2, :mixed
    bondratio::Real = 1.0 # only used when couplingtype is not :uniform
                          # for Honeycomb{:brickwall}, bondratio<1 is plaquette, bondratio>1 is dimmer
end

"""
    hamiltonian(model::J1J2J3)

Construct the two-site interaction term of the J1-J2-J3 Heisenberg model. The returned tensor `h[i,j,k,l]` represents the interaction between two spins, where `i,j` are the indices for the first spin and `k,l` are the indices for the second spin. The Hamiltonian is given by:
"""
function hamiltonian(model::J1J2J3)
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
