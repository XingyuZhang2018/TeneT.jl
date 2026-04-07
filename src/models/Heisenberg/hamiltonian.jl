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
    couplingtype::Symbol = :uniform # :uniform, :plaquette...
    bondratio = 1.0
end

"""
    hamiltonian(model::Heisenberg)

Return the Heisenberg Hamiltonian as a two-site operator.
When `ifrotate=true`, applies a sublattice rotation `U = 2Sy` so that the
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
        U = Sy * 2
        @tensor h_rot[i,j,k,l] := h[i,j,c,d] * U[k,c] * conj(U[l,d])
        h = h_rot
    end
    return real(h)
end

function hamiltonian(model::Heisenberg{Kagome{:merge}})
    @unpack Jx, Jy, Jz, S = model
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    d = size(Sx, 1)
    Id = Matrix{Float64}(I, d, d)
    h_H = Jx * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sx[3,9] * Sx[4,10] * Id[5,11] * Id[6,12]) + 
          Jy * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sy[3,9] * Sy[4,10] * Id[5,11] * Id[6,12]) + 
          Jz * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sz[3,9] * Sz[4,10] * Id[5,11] * Id[6,12]) +
          Jx * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sx[3,9] * Id[4,10] * Sx[5,11] * Id[6,12]) + 
          Jy * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sy[3,9] * Id[4,10] * Sy[5,11] * Id[6,12]) + 
          Jz * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sz[3,9] * Id[4,10] * Sz[5,11] * Id[6,12])

    h_V = Jx * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sx[3,9] * Sx[4,10] * Id[5,11] * Id[6,12]) +
          Jy * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sy[3,9] * Sy[4,10] * Id[5,11] * Id[6,12]) +
          Jz * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sz[3,9] * Sz[4,10] * Id[5,11] * Id[6,12]) +
          Jx * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Sx[2,8] * Id[3,9] * Sx[4,10] * Id[5,11] * Id[6,12]) +
          Jy * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Sy[2,8] * Id[3,9] * Sy[4,10] * Id[5,11] * Id[6,12]) +
          Jz * (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Sz[2,8] * Id[3,9] * Sz[4,10] * Id[5,11] * Id[6,12])

    return reshape(real(h_H),d^3,d^3,d^3,d^3), reshape(real(h_V),d^3,d^3,d^3,d^3)
end

function hamiltonian_onsite(model::Heisenberg{Kagome{:merge}})
    @unpack Jx, Jy, Jz, S = model
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    d = size(Sx, 1)
    Id = Matrix{Float64}(I, d, d)
    h = Jx * (@tensor out[1,2,3,4,5,6] := Sx[1,4] * Sx[2,5] * Id[3,6]) +
        Jy * (@tensor out[1,2,3,4,5,6] := Sy[1,4] * Sy[2,5] * Id[3,6]) +
        Jz * (@tensor out[1,2,3,4,5,6] := Sz[1,4] * Sz[2,5] * Id[3,6]) + 
        Jx * (@tensor out[1,2,3,4,5,6] := Id[1,4] * Sx[2,5] * Sx[3,6]) +
        Jy * (@tensor out[1,2,3,4,5,6] := Id[1,4] * Sy[2,5] * Sy[3,6]) +
        Jz * (@tensor out[1,2,3,4,5,6] := Id[1,4] * Sz[2,5] * Sz[3,6])

    return reshape(real(h),d^3,d^3)
end
