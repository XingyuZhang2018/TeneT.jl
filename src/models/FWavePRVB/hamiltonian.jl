export FWavePRVB

"""
    FWavePRVB{L<:AbstractLattice}

Ring exchange model on honeycomb lattice.
H = J1 Σ_NN S⃗ᵢ·S⃗ⱼ  +  K Σ_hexagons K₆

where K₆ = C₆ + C₆⁻¹ is the ring exchange (cyclic permutation) operator.
The f-wave pRVB state |f⟩ = (|K₁⟩ - |K₂⟩)/√2 is an eigenstate of K₆
with eigenvalue -2 (the minimum). With K>0, the +K·K₆ term favors
the f-wave ground state (E_f = -2K).

Set J1=0 for the pure ring exchange model.
"""
@kwdef mutable struct FWavePRVB{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Honeycomb(:brickwall)
    S::Real = 1/2
    J1::Real = 0.0
    K::Real = 1.0     # ring exchange coupling
    couplingtype::Symbol = :uniform
    bondratio::Real = 1.0
end

"""
    hamiltonian(model::FWavePRVB)

Returns the Heisenberg S⃗·S⃗ two-site operator for the J1 term.
The K₆ ring exchange term is handled directly in energy_value via contract_o_23.
"""
function hamiltonian(model::FWavePRVB)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    @tensor hx[i,j,k,l] := Sx[i,j] * Sx[k,l]
    @tensor hy[i,j,k,l] := Sy[i,j] * Sy[k,l]
    @tensor hz[i,j,k,l] := Sz[i,j] * Sz[k,l]
    return real(hx + hy + hz)
end
