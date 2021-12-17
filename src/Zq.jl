struct Z{Q} <: Number z::Int end
import Base.+,Base.-,Base.convert
+(x::Z{Q},y::Z{Q}) where Q = Z{Q}(mod(x.z+y.z,Q))
-(x::Z{Q},y::Z{Q}) where Q = Z{Q}(mod(x.z-y.z,Q))


import Base.*,Base./
*(x::Z{Q},y::Z{Q}) where Q = Z{Q}(mod(x.z*y.z,Q))
/(x::Z{Q},y::Z{Q}) where Q = Z{Q}(mod(x.z*y.z,Q))

convert(::Type{Int},x::Z) = Int(x)
convert(::Type{Z{Q}},x::Z{Q}) where Q = x
convert(::Type{Any},x::Z{Q}) where Q = x
Int(x::Z) = x.z


Q(z::Z) = typeof(z).parameters[1]
value(z::Z) = z.z

const VVZ{Q} = Vector{Vector{Z{Q}}}

"""
    A Tensor with Zq Symmetry.
    qspace is a array: Map a series of Quantum Number of an array of hash index
"""
struct ZqTensor 
    qspace::Tuple{Vararg{Tuple{Vararg{Int}}}} # ((q1,q2,q3),(q1,q2)...)
    dspace::Array{AbstractArray,1}

    hash_q::Array # [1,q1,q1*q2,q1*q2*q3.....]
end

function ZqTensor(qspace)
    qshape =  vcat(collect(map(collect,qspace))...)
    hash_q = vcat([1],[prod(qshape[1:i]) for i = 1:length(qshape)-1])
    dspace = Array{AbstractArray,1}(undef,prod(qshape))
    return ZqTensor(qspace,dspace,hash_q)
end

hash_qspace(zt::ZqTensor,qn::VVZ{Q}) where Q = sum(Int64.(vcat(qn...)).*zt.hash_q)+1

"""
    when contraction, we only need iteratively access QN in a single bond
"""
function getqnblock(zt::ZqTensor,qn::VVZ{Q}) where Q
    return zt.dspace[hash_qspace(zt,qn)]
end

function setqnblock!(zt::ZqTensor,qn::VVZ{Q},data::AbstractArray) where Q
    zt.dspace[hash_qspace(zt,qn)] = data
end



qdim = ((2,),(2,),(2,),(2,))
q = [[Z{2}(1)] for i = 1:4]
zt = ZqTensor(qdim)
hash_qspace(zt,q)
@show hash_qspace(zt,[[Z{2}(0)],[Z{2}(0)],[Z{2}(1)],[Z{2}(0)]])
setqnblock!(zt,q,rand(4,4))
getqnblock(zt,q)
