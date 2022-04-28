import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
using BitBasis
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!, axpby!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import Zygote: accum
export U1Array, U1reshape, U1reshapeinfo
export randU1, asU1Array, asArray
export dtr

# """
#     qn_conserving(T::Array)
# Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into qn conserving form
# ----
# The following is faster but rely on updates of CUDA.jl(available in master branch)
# function qn_conserving!(T::Union{Array,CuArray})
# 	bits = map(x -> Int(ceil(log2(x))), size(T))
#     T[map(x->sum(sum.(bitarray.((Tuple(x).-1) ,bits))) % 2 !== 0 ,CartesianIndices(T))].=0
#     return T
# end
# qn_conserving(T) = qn_conserving!(copy(T))
# """
# function qn_conserving(T::Union{Array,CuArray})
# 	s = size(T)
# 	p = zeros(s)
# 	bits = map(x -> Int(ceil(log2(x))), s)
# 	@inbounds @simd for index in CartesianIndices(T)
# 		i = Tuple(index) .- 1
# 		sum(sum.(bitarray.(i,bits))) % 2 == 0 && (p[index] = 1)
# 	end
# 	p = _arraytype(T)(p)

# 	return reshape(p.*T,s...)
# end

"""
    U1Array{T, N}

a struct to hold the N-order U1 tensors
- `qn`(`quantum number`): `N`-length Array
- `dir`(`out or in`): +1 or -1
- `tensor`: bulk tensor
- `size`  : original none-symetric array size
- `dims` size of `tensor`
- `division`: division location for reshape
"""
struct U1Array{T, N} <: AbstractSymmetricArray{T,N}
    qn::Vector{Vector{Int}}
    dir::Vector{Int}
    tensor::Vector{AbstractArray{T}}
    size::Tuple{Vararg{Int, N}}
    dims::Vector{Vector{Int}}
    division::Int
    function U1Array(qn::Vector{Vector{Int}}, dir::Vector{Int}, tensor::Vector{<:AbstractArray{T}}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int) where {T,N}
        new{T, N}(qn, dir, tensor, size, dims, division)
    end
end

size(A::U1Array) = A.size
size(A::U1Array, a) = size(A)[a]
getdir(A::U1Array) = A.dir
getq(A::U1Array) = unique(sum.(A.qn))
conj(A::U1Array) = U1Array(A.qn, -A.dir, map(conj, A.tensor), A.size, A.dims, A.division)
map(conj, A::U1Array) = conj(A)
norm(A::U1Array) = norm(A.tensor)

*(A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor * B, A.size, A.dims, A.division)
*(B::Number, A::U1Array{T,N}) where {T,N} = A * B
/(A::U1Array{T,N}, B::Number) where {T,N} = U1Array(A.qn, A.dir, A.tensor / B, A.size, A.dims, A.division)
broadcasted(*, A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(*, B::Number, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(/, A::U1Array, B::Number) = A / B

function +(A::U1Array, B::U1Array)
    if B.qn == A.qn
        U1Array(B.qn, B.dir, A.tensor + B.tensor, B.size, B.dims, B.division)
    else
        qn = intersect(A.qn, B.qn)
        tensor = A.tensor[indexin(qn, A.qn)] + B.tensor[indexin(qn, B.qn)]
        extraqn = setdiff(A.qn, B.qn)            # setdiff result is dependent on order
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            push!(tensor, A.tensor[indexin(extraqn, A.qn)]...)
        end
        extraqn = setdiff(B.qn, A.qn)
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            push!(tensor, B.tensor[indexin(extraqn, B.qn)]...)
        end

        dims = map(x -> collect(size(x)), tensor)
        U1Array(qn, A.dir, tensor, A.size, dims, A.division)
    end
end

function -(A::U1Array, B::U1Array)
    if B.qn == A.qn
        U1Array(B.qn, B.dir, A.tensor - B.tensor, B.size, B.dims, B.division)
    else
        atype = typeof(A.tensor[1])
        qn = intersect(A.qn, B.qn)
        tensor = Vector{atype}(A.tensor[indexin(qn, A.qn)] - B.tensor[indexin(qn, B.qn)])
        extraqn = setdiff(A.qn, B.qn)            # setdiff result is related to order
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            push!(tensor, A.tensor[indexin(extraqn, A.qn)]...)
        end
        extraqn = setdiff(B.qn, A.qn)
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            push!(tensor, -B.tensor[indexin(extraqn, B.qn)]...)
        end

        dims = map(x -> collect(size(x)), tensor)
        U1Array(qn, A.dir, tensor, A.size, dims, A.division)
    end
end

-(A::U1Array) = U1Array(A.qn, A.dir, map(-, A.tensor), A.size, A.dims, A.division)

CuArray(A::U1Array) = U1Array(A.qn, A.dir, map(CuArray, A.tensor), A.size, A.dims, A.division)
Array(A::U1Array) = U1Array(A.qn, A.dir, map(Array, A.tensor), A.size, A.dims, A.division)

function dot(A::U1Array, B::U1Array) 
    if A.qn == B.qn 
        dot(A.tensor, B.tensor)
    # elseif length(A.qn) > length(B.qn)
    #     # @warn "dot product of U1Array with different quantum numbers"
    #     # setdiff(B.qn, A.qn) !== [] && @show setdiff(B,A)
    #     exchangeind = indexin(B.qn, A.qn)
    #     dot(A.tensor[exchangeind], B.tensor)
    else
        # @warn "dot product of U1Array with different quantum numbers"
        commonqn = intersect(A.qn, B.qn)
        # @show commonqn A.qn B.qn
        # exchangeind = indexin(A.qn, B.qn)
        # @show indexin(commonqn, A.qn) indexin(commonqn, B.qn)
        commonqn == [] && return 0.0
        dot(A.tensor[indexin(commonqn, A.qn)], B.tensor[indexin(commonqn, B.qn)])
    end
end

function ≈(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    exchangeind = indexin(A.qn, B.qn)
    A.tensor ≈ B.tensor[exchangeind]
end

function ==(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    exchangeind = indexin(A.qn, B.qn)
    A.tensor == B.tensor[exchangeind]
end

function show(::IOBuffer, A::U1Array)
    println("particle number: \n", A.qn)
    println("direction: \n", A.dir)
    println("dims: \n", A.dims)
    println("tensor: \n", A.tensor)
end

"""
    maxq(D::Int)

give the maximum `q` value for a given dimensions `D`
"""
maxq(D::Int) = floor(Int, log2(D))


"""
    minuseven!(A)

return the negative of the array with even indices
"""
minuseven(A) = (A = Array{Int}(A); A[[i % 2 == 0 for i in 1:length(A)]] .*= -1; A)

"""
    bkdims = u1bulkdims(size::Int...)

distribute dims of different part dims of U1 tensor bulk by bits division
"""
function u1bulkdims(s::Int...)
    mq = maxq.(s)
    qn = map((s,q) -> [sum(bitarray(i - 1, q + 1)) for i = 1:s], s, mq)
    map((qn,mq) -> [sum(qn .== i) for i = 0:mq], qn, mq)
end

function randU1(atype, dtype, s...; dir, q::Vector{Int}=[0])
    L = length(s)
    bkdims = u1bulkdims(s...) # custom initial
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    mq = maxq.(s)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:mq for mq in mq))
        if sum(i.I .* dir) in q
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
            push!(qn, collect(i.I))
            push!(tensor, atype(rand(dtype, dims)))
        end
    end
    dims = map(x -> collect(size(x)), tensor)
    U1Array(qn, dir, tensor, s, dims, 1)
end

function zerosU1(atype, dtype, s...; dir, q::Vector{Int}=[0])
    L = length(s)
    bkdims = u1bulkdims(s...) # custom initial
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    mq = maxq.(s)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:mq for mq in mq))
        if sum(i.I .* dir) in q
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
            push!(qn, collect(i.I))
            push!(tensor, atype(zeros(dtype, dims)))
        end
    end
    dims = map(x -> collect(size(x)), tensor)
    U1Array(qn, dir, tensor, s, dims, 1)
end

zero(A::U1Array) = U1Array(A.qn, A.dir, map(zero, A.tensor), A.size, A.dims, A.division)

function IU1(atype, dtype, D; dir, q::Vector{Int}=[0])
    bkdims = u1bulkdims(D, D) # custom initial
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    mq = maxq.((D, D))
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:mq for mq in mq))
        if sum(i.I .* dir) in q
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:2)
            push!(qn, collect(i.I))
            push!(tensor, atype{dtype}(I, dims))
        end
    end
    dims = map(x -> collect(size(x)), tensor)
    U1Array(qn, dir, tensor, (D, D), dims, 1)
end

getindex(A::U1Array, index::CartesianIndex) = getindex(A, index.I...)
function getindex(A::U1Array{T,N}, index::Int...) where {T,N}
    bits = map(x -> ceil(Int, log2(x)), size(A))
    qn = collect(map((index, bits) -> sum(bitarray(index - 1, bits)), index, bits))
    # sum(qn.*Adir) != 0 && return 0.0
    ind = findfirst(x->x in [qn], A.qn)
    ind === nothing && return 0.0
    qlist = [U1selection(size(A, i)) for i = 1:N]
    position = [sum(qlist[i][qn[i]+1][1:index[i]]) for i in 1:N]
    CUDA.@allowscalar A.tensor[ind][position...]
end

setindex!(A::U1Array, x::Number, index::CartesianIndex) = setindex!(A, x, index.I...)
function setindex!(A::U1Array{T,N}, x::Number, index::Int...) where {T,N}
    bits = map(x -> ceil(Int, log2(x)), size(A))
    qn = collect(map((index, bits) -> sum(bitarray(index - 1, bits)), index, bits))
    ind = findfirst(x->x in [qn], A.qn)
    qlist = [U1selection(size(A, i)) for i = 1:N]
    position = [sum(qlist[i][qn[i]+1][1:index[i]]) for i in 1:N]
    CUDA.@allowscalar A.tensor[ind][position...] = x
end

function U1selection(maxs::Int)
    mq = maxq(maxs)
    q = [sum(bitarray(i-1, mq + 1)) for i = 1:maxs]
    [q .== i for i in 0:mq]
end

function asArray(A::U1Array{T,N}) where {T,N}
    atype = _arraytype(A.tensor[1])
    tensor = zeros(T, size(A))
    qn = A.qn
    qlist = [U1selection(size(A, i)) for i = 1:N]
    shift = 1
    for i in 1:length(qn)
        tensor[[qlist[j][abs(qn[i][j])+shift] for j = 1:N]...] = Array(A.tensor[i])
    end
    atype(tensor)
end

"""
p = getqn(s)

give the qn of length L
"""
function getqn(s, dir::Vector{Int}, q::Vector{Int}=[0])
    qm = maxq.(s)
    qn = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(0:qm for qm in qm))
        if sum(i.I .* dir) in q
            push!(qn, collect(i.I))
        end
    end
    qn
end

function deletezerobulk(A::U1Array)
    nozeroind = norm.(A.tensor) .> 1e-10
    U1Array(A.qn[nozeroind], A.dir, A.tensor[nozeroind], A.size, A.dims[nozeroind], A.division)
end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function asU1Array(A::AbstractArray{T,N}; dir::Vector{Int}, q::Vector{Int}=[0]) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [U1selection(size(A, i)) for i = 1:N]
    qn = getqn(size(A), dir, q)
    shift = 1
    tensor = [atype(Aarray[[qlist[j][abs(qn[i][j])+shift] for j = 1:N]...]) for i in 1:length(qn)]
    dims = map(x -> collect(size(x)), tensor)
    deletezerobulk(U1Array(qn, dir, tensor, size(A), dims, 1))
end

# # only for OMEinsum binary permutedims before reshape
permutedims(A::U1Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::U1Array, perm)
    length(perm) == 0 && return copy(A)
    qn = map(x -> x[collect(perm)], A.qn)
    tensor = map(x -> permutedims(x, perm), A.tensor)
    dims = map(x -> x[collect(perm)], A.dims)
    U1Array(qn, A.dir[collect(perm)], tensor, A.size[collect(perm)], dims, A.division)
end

reshape(A::U1Array, s::Tuple{Vararg{Int}}) = reshape(A, s...)
function reshape(A::U1Array{T,N}, s::Int...) where {T,N}
    Atensor = A.tensor
    div = 1
    if length(s) < N
        sizeA = size(A)
        p = sizeA[1]
        while p != s[1]
            div += 1
            p *= sizeA[div]
        end
        tensor = map((x, y) -> reshape(x, prod(y[1:div]), prod(y[div+1:end])), Atensor, A.dims)
        return U1Array(A.qn, A.dir, tensor, A.size, A.dims, div)
    else
        tensor = map((x, y) -> reshape(x, y...), Atensor, A.dims)
        return U1Array(A.qn, A.dir, tensor, A.size, A.dims, A.division)
    end
end

"""
    *(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,TB,NA,NB}

core code for U1Array product
"""
function *(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,TB,NA,NB}
    qn = Vector{Vector{Int}}()
    dims = Vector{Vector{Int}}()
    # qn_para = Vector{Vector{Vector{Int}}}()
    # dims_para = Vector{Vector{Vector{Int}}}()
    atype = _arraytype(B.tensor[1])
    T = TA == ComplexF64 || TB == ComplexF64 ? ComplexF64 : Float64
    tensor = Vector{atype{T}}()
    # tensor_para = Vector{Vector{atype{T}}}()
    divA, divB = A.division, B.division

    Adir = getdir(A)[divA+1:end]
    Bdir = getdir(B)[1:divB]
    sum(Adir .+ Bdir) !== 0 && throw(Base.error("U1Array product: out and in direction not match, expect: $(-Adir), got: $(Bdir)"))

    for p in unique(map(qn -> sum(qn[divA+1:end] .* A.dir[divA+1:end]), A.qn))
        u1bulktimes!(qn, tensor, dims, A, B, p)
    end
    # for _ in 1:Threads.nthreads()
    #     push!(qn_para, Vector{Vector{Int}}())
    #     push!(dims_para, Vector{Vector{Int}}())
    #     push!(tensor_para, Vector{atype{T}}())
    # end
    
    # Threads.@threads for p in unique(map(qn -> sum(qn[divA+1:end] .* A.dir[divA+1:end]), A.qn))
    #     pi = Threads.threadid()
    #     u1bulktimes!(qn_para[pi], tensor_para[pi], dims_para[pi], A, B, p)
    # end
    # qn = vcat(qn_para...)
    # dims = vcat(dims_para...)
    # tensor = vcat(tensor_para...)

    qn == [[]] && return Array(tensor[1])[]
    U1Array(qn, [A.dir[1:divA]..., B.dir[divB+1:end]...], tensor, (size(A)[1:divA]..., size(B)[divB+1:end]...), dims, divA)

    # Aqn, Atensor = A.qn, A.tensor
    # Bqn, Btensor = B.qn, B.tensor
    # Adims, Bdims = A.dims, B.dims
    # LA, LB = length(Aqn), length(Bqn)

    # Threads.@threads for k in 1:LA*LB
    #     i,j = ktoij(k, LA, LB)
    #     if Aqn[i][divA+1:end] == Bqn[j][1:divB]
    #         pi = Threads.threadid()
    #         push!(qn_para[pi], [Aqn[i][1:divA]; Bqn[j][divB+1:end]])
    #         push!(dims_para[pi], [Adims[i][1:divA]; Bdims[j][divB+1:end]])
    #         push!(tensor_para[pi], Atensor[i] * Btensor[j])
    #     end
    # end

    # qn = vcat(qn_para...)
    # dims = vcat(dims_para...)
    # tensor = vcat(tensor_para...)

    # uqn = unique(qn)
    # udims = Vector{Vector{Int}}()
    # utensor = Vector{atype{T}}()
    # for uqn in uqn
    #     ind = [uqn] .== qn
    #     push!(utensor, +(tensor[ind]...))
    #     push!(udims, dims[ind][1])
    # end
    # uqn == [[]] && return Array(utensor[1])[]

    # U1Array(uqn, [A.dir[1:divA]..., B.dir[divB+1:end]...], utensor, (size(A)[1:divA]..., size(B)[divB+1:end]...), udims, divA)
end

function no_nothing_col(index)
    indexcol = Int[]
    for i in 1:size(index,1)
        for j in 1:size(index,2)
            if index[i, j] !== nothing
                push!(indexcol, index[i, j])
                break
            end
        end
    end
    indexcol
end

function no_nothing_row(index)
    indexrow = Int[]
    for j in 1:size(index,2)
        for i in 1:size(index,1)
            if index[i, j] !== nothing
                push!(indexrow, index[i, j])
                break
            end
        end
    end
    indexrow
end

"""
    u1bulktimes!(qn, tensor, A, B, p)

fill into different particle number,  then dispatch to result tensor after product
"""
function u1bulktimes!(qn, tensor, dims, A, B, p)
    Aqn, Atensor = A.qn, A.tensor
    Bqn, Btensor = B.qn, B.tensor
    Adims, Bdims = A.dims, B.dims
    divA, divB = A.division, B.division
    atype = _arraytype(Btensor[1])
    etype = eltype(Btensor[1])

    ind_A = findall(x->sum(x[divA+1:end] .* A.dir[divA+1:end]) == p, Aqn)
    matrix_j = intersect(map(x->x[divA+1:end], Aqn[ind_A]), map(x->x[1:divB], Bqn))
    ind_A = findall(x->x[divA+1:end] in matrix_j, Aqn)
    matrix_i = unique(map(x->x[1:divA], Aqn[ind_A]))
    ind_B = findall(x->x[1:divB] in matrix_j, Bqn)
    ind_B == [] && return
    matrix_k = unique(map(x->x[divB+1:end], Bqn[ind_B]))

    # @show Aqn Bqn matrix_i matrix_j ind_A ind_B matrix_k
    index = [findfirst(x->x in [[i; j]], Aqn) for i in matrix_i, j in matrix_j]
    # @show index
    if nothing in index
        indexcol = no_nothing_col(index)
        indexrow = no_nothing_row(index)
    else
        indexcol = index[:, 1]
        indexrow = index[1, :]
    end

    oribulkidims = map(ind -> Adims[ind][1:divA], indexcol)
    bulkidims = map(ind -> size(Atensor[ind], 1), indexcol)
    bulkjdims = map(ind -> size(Atensor[ind], 2), indexrow)
    # Amatrix = hvcat(ntuple(i->length(bulkjdims), length(bulkidims)), Atensor[index']...)
    Amatrix = atype <: Array ? zeros(etype, sum(bulkidims), sum(bulkjdims)) : CUDA.zeros(etype, sum(bulkidims), sum(bulkjdims))
    # @show size(Amatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        # println(sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), ", ", sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), " ", index[i, j])
        index[i, j] !== nothing && (Amatrix[sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])] .= Atensor[index[i, j]])
    end

    index = [findfirst(x->x in [[j; k]], Bqn) for j in matrix_j, k in matrix_k]
    oribulkkdims = []
    bulkkdims = Vector{Int}()
    indexrow = nothing in index ? no_nothing_row(index) : index[1, :]
    oribulkkdims = map(ind -> Bdims[ind][divB+1:end], indexrow)
    bulkkdims = map(ind -> size(Btensor[ind], 2), indexrow)
    # Bmatrix = hvcat(ntuple(i->length(bulkkdims), length(bulkjdims)), Btensor[index']...)
    Bmatrix = atype <: Array ? zeros(etype, sum(bulkjdims), sum(bulkkdims)) : CUDA.zeros(etype, sum(bulkjdims), sum(bulkkdims))
    # @show size(Bmatrix)
    for j in 1:length(matrix_j), k in 1:length(matrix_k)
        # println(sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), ", ", sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k]), " ", index[j, k])
        index[j, k] !== nothing && (Bmatrix[sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])] .= Btensor[index[j, k]])
    end
    
    C = Amatrix * Bmatrix

    for i in 1:length(matrix_i), k in 1:length(matrix_k)
        push!(qn, [matrix_i[i]; matrix_k[k]])
        push!(dims, [oribulkidims[i]; oribulkkdims[k]])
        idim, kdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])
        # println(idim, ", ", kdim)
        push!(tensor, C[idim, kdim])
    end
end


# # for OMEinsum contract to get number
# # vec(A::U1Array) = A

function transpose(A::U1Array)
    tensor = map(transpose, A.tensor)
    U1Array(A.qn, A.dir, tensor, A.size, A.dims, 0)
end

function tr(A::U1Array{T,N}) where {T,N}
    qn = A.qn
    tensor = A.tensor
    half = Int(length(qn[1])/2)
    s = 0.0
    @inbounds @simd for i in 1:length(qn)
        qn[i][1:half] == qn[i][half+1:end] && (s += tr(tensor[i]))
    end
    s
end

function _compactify!(y, x::U1Array, indexer)
    x = asArray(Array(x))
    @inbounds @simd for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
end

broadcasted(*, A::U1Array, B::Base.RefValue) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(*, B::Base.RefValue, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)

# for ein"abab ->"(A)[]
function dtr(A::U1Array{T,N}) where {T,N}
    qn = A.qn
    tensor = A.tensor
    s = 0.0
    @inbounds @simd for i in 1:length(qn)
        qn[i][1] == qn[i][3] && qn[i][2] == qn[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end


# for Zygote compatibility
accum(A::U1Array, B::U1Array...) = +(A, B...)

# for KrylovKit compatibility
rmul!(A::U1Array, B::Number) = (map(x -> rmul!(x, B), A.tensor); A)

similar(A::U1Array) = U1Array(map(copy, A.qn), copy(A.dir), map(similar, A.tensor), A.size, A.dims, A.division)
similar(A::U1Array, atype) = U1Array(map(copy, A.qn), copy(A.dir), map(x -> atype(similar(x)), A.tensor), A.size, A.dims, A.division)
diag(A::U1Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))
copy(A::U1Array{T,N}) where {T,N} = U1Array(map(copy, A.qn), copy(A.dir), map(copy, A.tensor), A.size, A.dims, A.division)

function mul!(Y::U1Array, A::U1Array, B::Number)
    if Y.qn == A.qn
        map((Y, A) -> mul!(Y, A, B), Y.tensor, A.tensor)
    else
        # @warn "mul!(Y, A, B) : length(A.qn) !== length(B.qn)"
        # exchangeind = indexin(A.qn, Y.qn)  # A.qn is a subset of Y.qn
        # map((Y, A) -> mul!(Y, A, B), Y.tensor[exchangeind], A.tensor)
        Y = A*B
    end
    # @show length.([Y.qn, Y.tensor])
    Y
end

function axpy!(α::Number, X::U1Array, Y::U1Array)
    if Y.qn == X.qn
        map((x,y) -> axpy!(α, x, y), X.tensor, Y.tensor)
    else
        # inter = intersect(X.qn, Y.qn)
        # map((x,y) -> axpy!(α, x, y), X.tensor[indexin(inter, X.qn)], Y.tensor[indexin(inter, Y.qn)])
        # extraqn = setdiff(X.qn, Y.qn)
        # if length(extraqn) !== 0
        #     # @show 3
        #     push!(Y.qn, extraqn...)
        #     ind = indexin(extraqn, X.qn)
        #     push!(Y.tensor, α * X.tensor[ind]...)
        #     push!(Y.dims, X.dims[ind]...)
        # end
        Y = α * X + Y
    end
    # @show 33333333333 length.([Y.qn, Y.tensor]) length.([X.qn, X.tensor])
    return Y
end

function axpby!(α::Number, x::U1Array, β::Number, y::U1Array)
    if x.qn == y.qn
        map((x,y) -> axpby!(α, x, β, y), x.tensor, y.tensor)
    else
        # length(x.qn) !== length(y.qn) && @warn "axpby!(x, β, y) is not implemented for x.qn != y.qn"
        y = α * x + β * y
        # exchangeind = indexin(y.qn, x.qn)
        # map((x,y) -> axpby!(α, x, β, y), x.tensor[exchangeind], y.tensor)
    end
    return y
end

# # for leftorth and rightorth compatibility
Diagonal(A::U1Array) = U1Array(A.qn, A.dir, map(Diagonal, A.tensor), A.size, A.dims, A.division)
sqrt(A::U1Array) = U1Array(A.qn, A.dir, map(x->sqrt.(x), A.tensor), A.size, A.dims, A.division)
broadcasted(sqrt, A::U1Array) = sqrt(A)
function lmul!(A::U1Array, B::U1Array)
    C = A*B
    for i = 1:length(B.qn)
        B.tensor[i] = C.tensor[i]
        B.qn[i] = C.qn[i]
        B.dims[i] = C.dims[i]
    end
    return B
end

# only for order-three tensor's qr and lq
function qrpos!(A::U1Array{T,N}) where {T,N}
    Qqn = Vector{Vector{Int}}()
    Rqn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Qtensor = Vector{atype{T}}()
    Rtensor = Vector{atype{T}}()

    for p in unique(map(x->sum(x[A.division+1:end] .* A.dir[A.division+1:end]), A.qn))
        # @show p
        u1bulkQR!(Qqn, Qtensor, Rqn, Rtensor, A, p)
    end
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qqn, A.qn)
    U1Array(Qqn, A.dir, Qtensor, Asize, Adims[exchangeind], A.division), U1Array(Rqn, [-A.dir[end], A.dir[end]], Rtensor, (Asize[end], Asize[end]), map(x -> [size(x)...], Rtensor), 1)
end

function u1bulkQR!(Qqn, Qtensor, Rqn, Rtensor, A, p)
    Atensor = A.tensor
    Aqn = A.qn
    Adiv = A.division

    ind_A = findall(x->sum(x[Adiv+1:end] .* A.dir[Adiv+1:end]) == p, Aqn)
    matrix_j = unique(map(x->x[Adiv+1:end], Aqn[ind_A]))
    matrix_i = unique(map(x->x[1:Adiv], Aqn[ind_A]))

    ind = [findfirst(x->x in [[i; matrix_j[1]]], Aqn) for i in matrix_i]
    Amatrix = vcat(Atensor[ind]...)
    bulkidims = [size(Atensor[i],1) for i in ind]
    bulkjdims = [size(Amatrix, 2)]

    Q, R = qrpos!(Amatrix)
    for i in 1:length(matrix_i)
        push!(Qqn, [matrix_i[i]; matrix_j[1]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), 1:sum(bulkjdims[1])
        push!(Qtensor, Q[idim, jdim])
    end
    
    push!(Rqn, [matrix_j[1]; matrix_j[1]])
    idim, jdim = 1:sum(bulkjdims[1]), 1:sum(bulkjdims[1])
    push!(Rtensor, R[idim, jdim])
end

function lqpos!(A::U1Array{T,N}) where {T,N}
    Lqn = Vector{Vector{Int}}()
    Qqn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Ltensor = Vector{atype{T}}()
    Qtensor = Vector{atype{T}}()

    for p in unique(map(x->x[1] * A.dir[1], A.qn))
        u1bulkLQ!(Lqn, Ltensor, Qqn, Qtensor, A, p)
    end
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qqn, A.qn)
    U1Array(Lqn, [A.dir[1], -A.dir[1]], Ltensor, (Asize[1], Asize[1]), map(x -> [size(x)...], Ltensor), 1), U1Array(Qqn, A.dir, Qtensor, Asize, Adims[exchangeind], A.division)
end

function u1bulkLQ!(Lqn, Ltensor, Qqn, Qtensor, A, p)
    Atensor = A.tensor
    Aqn = A.qn
    Adiv = A.division

    ind_A = findall(x->x[1] * A.dir[1] == p, Aqn)
    matrix_j = unique(map(x->x[Adiv+1:end], Aqn[ind_A]))
    matrix_i = unique(map(x->x[1], Aqn[ind_A]))

    ind = [findfirst(x->x in [[matrix_i[1]; j]], Aqn) for j in matrix_j]
    Amatrix = hcat(Atensor[ind]...)
    bulkidims = [size(Amatrix, 1)]
    bulkjdims = [size(Atensor[i], 2) for i in ind]
    
    L, Q = lqpos!(Amatrix)

    push!(Lqn, [matrix_i[1]; matrix_i[1]])
    idim, jdim = 1:sum(bulkidims[1]), 1:sum(bulkidims[1])
    push!(Ltensor, L[idim, jdim])
    for j in 1:length(matrix_j)
        push!(Qqn, [matrix_i[1]; matrix_j[j]])
        idim, jdim = 1:sum(bulkidims[1]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
end

# # for ' in ACCtoALAR of VUMPS
function adjoint(A::U1Array{T,N}) where {T,N}
    div = A.division 
    qn = map(x->x[[div+1:end;1:div]], A.qn)
    tensor = map(adjoint, A.tensor)
    dims = map(x -> x[[div+1:end;1:div]], A.dims)
    U1Array(qn, -A.dir[[div+1:end;1:div]], tensor, A.size[[div+1:end;1:div]], dims, N - div)
end

# only for U1 Matrix
function sysvd!(A::U1Array{T,N}) where {T,N}
    # Atensor = asArray(A)
    # Utensor, Stensor, Vtensor = sysvd!(Atensor)
    # dir = getdir(A)
    # U = asU1Array(Utensor; dir = dir, q=collect(-2:2))
    # S = asU1Array(Diagonal(Stensor); dir = dir, q=[0])
    # V = asU1Array(Vtensor; dir = dir, q=collect(-2:2))
    # return U, S, V
    tensor = A.tensor
    qn = A.qn
    div = A.division
    atype = _arraytype(tensor[1])
    Utensor = Vector{atype{T}}()
    Stensor = Vector{atype{T}}()
    Vtensor = Vector{atype{T}}()
    @inbounds @simd for t in tensor
        U, S, V = sysvd!(t)
        push!(Utensor, U)
        push!(Stensor, S)
        push!(Vtensor, V)
    end
    Nm = map(x->min(x...), A.dims)
    N1 = map((x, y) -> [x[1], y], A.dims, Nm)
    N2 = map((x, y) -> [y, x[2]], A.dims, Nm)
    Asize = A.size
    sm = min(Asize...)
    U1Array(qn, A.dir, Utensor, (Asize[1], sm), N1, div), U1Array(qn, A.dir, Stensor, (sm, sm), [[Nm[i], Nm[i]] for i in 1:length(qn)], div), U1Array(qn, A.dir, Vtensor, (sm, Asize[2]), N2, div)
end

"""
    U1reshape(A::U1Array{T, N}, a::Int...) where {T, N}

U1reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`, and the high-oreder U1tensor is from randU1 or zerosU1 function.
"""
U1reshape(A::U1Array, s::Tuple{Vararg{Int}}; kwarg...) = U1reshape(A, s...; kwarg...)
function U1reshape(A::U1Array{T, N}, s::Int...; reinfo = nothing) where {T, N}
    atype = _arraytype(A.tensor[1])
    if N > length(s)
        cA = zerosU1(Array, ComplexF64, A.size...; dir = getdir(A))
        qndiff = setdiff(cA.qn, A.qn)
        supind = indexin(qndiff, cA.qn)
        Aqn = [A.qn; cA.qn[supind]]
        Atensor = [A.tensor; cA.tensor[supind]]
        exchangeind = indexin(cA.qn, Aqn)
        Aqn = cA.qn
        Adims = cA.dims
        Atensor = Atensor[exchangeind]
        div = division(s, size(A))
        reqn = [[sum(p[d] .* A.dir[d]) for d in div] for p in Aqn]
        redims = [[prod(dims[d]) for d in div] for dims in Adims]
        retensor = [reshape(t, s...) for (t, s) in zip(map(Array, Atensor), redims)]
        ureqn = unique(reqn)
        retensors = Vector{atype{T}}()
        inddims = u1bulkdims(size(A)...)   # only correct for random initial bits division U1Array
        choosesilces = [[] for _ in 1:length(ureqn)]
        chooseinds = [[] for _ in 1:length(ureqn)]
        for i in 1:length(ureqn)
            q = ureqn[i]
            bulkind = findall(x->x in [q], reqn)
            oriqn = Aqn[bulkind]
        
            indqnfrom = [unique(map(x->x[div], oriqn)) for div in div]
            rebulkdims = [[prod(map((x,y)->x[abs(y)+1], inddims[div[i]], indqnfrom)) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
            # @show indqnfrom
            # indqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
            # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
            silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
            tensor = atype(zeros(T, map(sum, rebulkdims)...))
            for j in 1:length(bulkind)
                chooseind = [indexin([oriqn[j][div[i]]], indqnfrom[i]) for i in 1:length(div)]
                choosesilce = map((s,i)->s[i...], silce, chooseind)
                tensor[choosesilce...] = retensor[bulkind[j]]
                push!(choosesilces[i], choosesilce)
                push!(chooseinds[i], bulkind[j])
            end
            push!(retensors, tensor)
        end
        dims = map(x -> collect(size(x)), retensors)
        dir = [A.dir[d][end] for d in div]     # last dir of reshape
        deletezerobulk(U1Array(map(qn->qn .* dir, ureqn), dir, map(atype, retensors), s, dims, 1)), (choosesilces, chooseinds, Aqn, A.dir, Adims)
    else
        choosesilces, chooseinds, reqn, redir, redims = reinfo
        retensors = Array{Array,1}(undef, sum(length.(chooseinds)))
        div = division(size(A), s)
        ureqn = unique([[sum(p[d] .* redir[d]) for d in div] for p in reqn])
        exchangeind = indexin(ureqn, map(qn->qn .* A.dir, A.qn))
        # @show ureqn A.qn exchangeind
        Atensor = A.tensor[exchangeind]
        for i in 1:length(choosesilces)
            for j in 1:length(choosesilces[i])
                retensors[chooseinds[i][j]] = reshape(Array(Atensor[i][choosesilces[i][j]...]), redims[chooseinds[i][j]]...)
            end
        end
        dims = map(x -> collect(size(x)), retensors)
        deletezerobulk(U1Array(reqn, redir, map(atype, retensors), s, dims, 1)), (choosesilces, chooseinds, reqn, redims)
    end
end

function U1reshapeinfo(s, sizeA, dir)
    length(sizeA) < length(s) && throw(Base.error("$sizeA must be longer than $s"))
    div = division(s, sizeA)
    A = zerosU1(Array, Float64, sizeA...; dir = dir)
    reqn = [[sum(p[d] .* dir[d]) for d in div] for p in A.qn]
    ureqn = unique(reqn)
    inddims = u1bulkdims(sizeA...)   # only correct for bits division U1Array
    choosesilces = [[] for _ in 1:length(ureqn)]
    chooseinds = [[] for _ in 1:length(ureqn)]
    for i in 1:length(ureqn)
        q = ureqn[i]
        bulkind = findall(x->x in [q], reqn)
        oriqn = A.qn[bulkind]
    
        indqnfrom = [unique(map(x->x[div], oriqn)) for div in div]
        rebulkdims = [[prod(map((x,y)->x[abs(y)+1], inddims[div[i]], indqnfrom)) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
        # @show indqnfrom
        # indqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
        # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
        silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
        for j in 1:length(bulkind)
            chooseind = [indexin([oriqn[j][div[i]]], indqnfrom[i]) for i in 1:length(div)]
            choosesilce = map((s,i)->s[i...], silce, chooseind)
            push!(choosesilces[i], choosesilce)
            push!(chooseinds[i], bulkind[j])
        end
    end
    choosesilces, chooseinds, A.qn, dir, A.dims
end