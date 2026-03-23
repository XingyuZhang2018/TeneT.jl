"""
    Heisenberg{L}(; lattice, S, Jx, Jy, Jz)

Heisenberg model with couplings `Jx`, `Jy`, `Jz` on a given lattice.
"""
@kwdef struct Heisenberg{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
end

"""
    hamiltonian(model::Heisenberg)

Return the Heisenberg Hamiltonian as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = model.Jx * (@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) +
        model.Jy * (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
        model.Jz * (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
    return real(h)
end
