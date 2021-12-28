using VUMPS
using VUMPS: _arraytype
using OMEinsum

const isingβc = log(1+sqrt(2))/2

abstract type HamiltonianModel end

struct Ising <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

"""
    model_tensor(model::Ising)
return the  `MT <: HamiltonianModel` bulktensor at inverse temperature `β` for  two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising; atype = Array)
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = ComplexF64[-1. 1;1 -1]
    w = exp.(- β * ham)
    wsq = sqrt(w)
    m = atype(ein"ia,ib,ic,id -> abcd"(wsq, wsq, wsq, wsq))
    reshape([m for i=1:Ni*Nj], Ni, Nj)
end

"""
    mag_tensor(::MT)
return the  `MT <: HamiltonianModel` the operator for the magnetisation at inverse temperature `β` for a two-dimensional
square lattice tensor-network. 
"""
function mag_tensor(model::Ising; atype = Array)
    Ni, Nj, β = model.Ni, model.Nj, model.β
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    atype == Z2tensor ? (m = tensor2Z2tensor(m)) : (m = atype(m))
    reshape([m for i=1:Ni*Nj], Ni, Nj)
end

"""
    energy_tensor(model::Ising)
return the  `MT <: HamiltonianModel` the operator for the energy at inverse temperature `β` for a two-dimensional
    square lattice tensor-network. 
"""
function energy_tensor(model::Ising; atype = Array)
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = ComplexF64[-1 1;1 -1]
    w = exp.(-β .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = wsq^(-1)
    e = atype(ein"ai,im,bm,cm,dm -> abcd"(wsqi,we,wsq,wsq,wsq) + ein"am,bi,im,cm,dm -> abcd"(wsq,wsqi,we,wsq,wsq) + 
        ein"am,bm,ci,im,dm -> abcd"(wsq,wsq,wsqi,we,wsq) + ein"am,bm,cm,di,im -> abcd"(wsq,wsq,wsq,wsqi,we)) / 2
    reshape([e for i=1:Ni*Nj], Ni, Nj)
end