"""
    Kagome{L}(; lattice, S, J1, J2)

Kagome model on a given lattice. Has special `hamiltonian_onsite`,
`hamiltonian_right`, and `hamiltonian_down` functions.
"""
@kwdef struct Kagome{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Kagome()
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.0
end

function hamiltonian_onsite(model::Kagome)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    d = size(Sx, 1)
    Id = Matrix{Float64}(I, d, d)
    h = (@tensor out[1,2,3,4,5,6] := Sx[1,4] * Sx[2,5] * Id[3,6]) +
        (@tensor out[1,2,3,4,5,6] := Sy[1,4] * Sy[2,5] * Id[3,6]) +
        (@tensor out[1,2,3,4,5,6] := Sz[1,4] * Sz[2,5] * Id[3,6]) +
        (@tensor out[1,2,3,4,5,6] := Id[1,4] * Sx[2,5] * Sx[3,6]) +
        (@tensor out[1,2,3,4,5,6] := Id[1,4] * Sy[2,5] * Sy[3,6]) +
        (@tensor out[1,2,3,4,5,6] := Id[1,4] * Sz[2,5] * Sz[3,6])

    return reshape(real(h),d^3,d^3)
end

function hamiltonian_right(model::Kagome)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    d = size(Sx, 1)
    Id = Matrix{Float64}(I, d, d)
    h = (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sx[3,9] * Sx[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sy[3,9] * Sy[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sz[3,9] * Sz[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sx[3,9] * Id[4,10] * Sx[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sy[3,9] * Id[4,10] * Sy[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sz[3,9] * Id[4,10] * Sz[5,11] * Id[6,12])

    return reshape(real(h),d^3,d^3,d^3,d^3)
end

function hamiltonian_down(model::Kagome)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    d = size(Sx, 1)
    Id = Matrix{Float64}(I, d, d)
    h = (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sx[3,9] * Sx[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sy[3,9] * Sy[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Id[2,8] * Sz[3,9] * Sz[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Sx[2,8] * Id[3,9] * Sx[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Sy[2,8] * Id[3,9] * Sy[4,10] * Id[5,11] * Id[6,12]) +
        (@tensor out[1,2,3,7,8,9,4,5,6,10,11,12] := Id[1,7] * Sz[2,8] * Id[3,9] * Sz[4,10] * Id[5,11] * Id[6,12])

    return reshape(real(h),d^3,d^3,d^3,d^3)
end
