using TeneT
using TeneT: _arraytype
using OMEinsum
using Zygote

const isingβc = log(1+sqrt(2))/2

abstract type HamiltonianModel end

struct Ising <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β
end

"""
    model_tensor(model::Ising, type)

return the  `MT <: HamiltonianModel` `type` tensor at inverse temperature `β` for  two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)

    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising, ::Val{:mag})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising, ::Val{:energy})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = ComplexF64[-1 1;1 -1]
    w = exp.(-β .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = wsq^(-1)
    e = (ein"ai,im,bm,cm,dm -> abcd"(wsqi,we,wsq,wsq,wsq) + ein"am,bi,im,cm,dm -> abcd"(wsq,wsqi,we,wsq,wsq) + 
        ein"am,bm,ci,im,dm -> abcd"(wsq,wsq,wsqi,we,wsq) + ein"am,bm,cm,di,im -> abcd"(wsq,wsq,wsq,wsqi,we)) / 2
    M = Zygote.Buffer(e, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = e
    end
    return copy(M)
end


