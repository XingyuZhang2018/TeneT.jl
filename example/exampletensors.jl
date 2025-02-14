using TeneT
using TeneT: _arraytype, rightenv, rightCenv
using OMEinsum
using Zygote
using Parameters

const isingβc = log(1+sqrt(2))/2

abstract type HamiltonianModel end

struct Ising <: HamiltonianModel 
    β::Float64
end

"""
    model_tensor(model::Ising, type)

return the  `MT <: HamiltonianModel` `type` tensor at inverse temperature `β` for  two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising, ::Val{:bulk})
    β = model.β
    ham = Zygote.@ignore ComplexF64[-1. 1;1 -1]
    w = exp.(- β * ham)
    wsq = sqrt(w)

    m = ein"ia,ib,ic,id -> abcd"(wsq, wsq, wsq, wsq)
    return m
end

function model_tensor(model::Ising, ::Val{:mag})
    β = model.β
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))

    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    return m 
end

function model_tensor(model::Ising, ::Val{:energy})
    β = model.β
    ham = ComplexF64[-1 1;1 -1]
    w = exp.(-β .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = wsq^(-1)
    e = (ein"ai,im,bm,cm,dm -> abcd"(wsqi,we,wsq,wsq,wsq) + ein"am,bi,im,cm,dm -> abcd"(wsq,wsqi,we,wsq,wsq) + 
        ein"am,bm,ci,im,dm -> abcd"(wsq,wsq,wsqi,we,wsq) + ein"am,bm,cm,di,im -> abcd"(wsq,wsq,wsq,wsqi,we)) / 2
    return e 
end


