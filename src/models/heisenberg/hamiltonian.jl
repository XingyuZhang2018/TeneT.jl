export Heisenberg

"""
    Heisenberg{L}(; lattice, S, Jx, Jy, Jz)

Heisenberg model with couplings `Jx`, `Jy`, `Jz` on a given lattice.
"""
@kwdef mutable struct Heisenberg{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
    ifrotate::Bool = true
end


"""
    hamiltonian(model::Heisenberg)

Return the Heisenberg Hamiltonian as a two-site operator.
When `ifrotate=true`, applies a sublattice rotation `U = 2Sx` so that the
antiferromagnetic Néel order becomes a uniform state, enabling efficient
optimization with a single-site unit cell.
"""
function hamiltonian(model::Heisenberg)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = model.Jx * (@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) +
        model.Jy * (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
        model.Jz * (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
    if model.ifrotate
        U = Sx * 2
        @tensor h_rot[i,j,k,l] := h[i,j,c,d] * U[k,c] * conj(U[l,d])
        h = h_rot
    end
    return real(h)
end
