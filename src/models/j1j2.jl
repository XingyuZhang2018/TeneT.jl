"""
    J1J2{L}(; lattice, S, J1, J2)

J1-J2 model on a given lattice.
"""
@kwdef struct J1J2{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.0
    ifrotate::Bool = true
end

function hamiltonian(model::J1J2)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = (@tensor out[i,j,k,l] := Sx[i,j] * Sx[k,l]) +
        (@tensor out[i,j,k,l] := Sy[i,j] * Sy[k,l]) +
        (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l])
    if model.ifrotate
        U = Sx * 2
        @tensor h_rot[i,j,k,l] := h[i,j,c,d] * U[k,c] * conj(U[l,d])
        h = h_rot
    end
    return real(h)
end
