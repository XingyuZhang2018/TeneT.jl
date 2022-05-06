import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, conj!, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
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
- `dir`(`out or in`): +1 or -1
- `qn`(`quantum number`): `N`-length Array
- `dims`(`backend dimensions`): `N`-length Array
- `qndims` qn of bulk tensor
- `bkdims` size of bulk tensor
- `tensor`: bulk tensor
- `division`: division location for reshape
"""
struct U1Array{T, N} <: AbstractSymmetricArray{T,N}
    dir::Vector{Int}
    qn::Vector{Vector{Int}}
    dims::Vector{Vector{Int}}
    bkqn::Vector{Vector{Int}}
    bkdims::Vector{Vector{Int}}
    tensor::Vector{AbstractArray{T}}
    division::Int
    function U1Array(dir::Vector{Int}, qn::AT, dims::AT, bkqn::AT, bkdims::AT, tensor::Vector{<:AbstractArray{T}}, division::Int) where {T, AT <: Vector{Vector{Int}}}
        N = length(dir)
        new{T, N}(dir, qn, dims, bkqn, bkdims, tensor, division)
    end
end

function show(::IOBuffer, A::U1Array)
    println("direction: \n", A.dir)
    println("quantum number: \n", A.qn)
    println("dims: \n", A.dims)
    println("bulkqn: \n", A.bkqn)
    println("bulkdims: \n", A.bkdims)
    println("tensor: \n", A.tensor)
end

function getbkqn(dir, qn; q = [0])
    L = length(dir)
    bkqn = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(qn)))
        bkqni = [qn[j][i.I[j]] for j in 1:L]
        if sum(bkqni .* dir) in q
            push!(bkqn, bkqni)
        end
    end
    bkqn
end

function randU1(atype, dtype, dir, qn, dims; q::Vector{Int}=[0])
    L = length(dir)
    bkqn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    bkdims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(qn)))
        bkqni = [qn[j][i.I[j]] for j in 1:L]
        if sum(bkqni .* dir) in q
            bulkdims = [dims[j][i.I[j]] for j in 1:L]
            push!(bkqn, bkqni)
            push!(bkdims, bulkdims)
            push!(tensor, atype(rand(dtype, bulkdims...)))
        end
    end
    U1Array(dir, qn, dims, bkqn, bkdims, tensor, 1)
end

function zerosU1(atype, dtype, dir, qn, dims; q::Vector{Int}=[0])
    L = length(dir)
    bkqn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    bkdims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(qn)))
        bkqni = [qn[j][i.I[j]] for j in 1:L]
        if sum(bkqni .* dir) in q
            bulkdims = [dims[j][i.I[j]] for j in 1:L]
            push!(bkqn, bkqni)
            push!(bkdims, bulkdims)
            push!(tensor, atype(zeros(dtype, bulkdims...)))
            
        end
    end
    U1Array(dir, qn, dims, bkqn, bkdims, tensor, 1)
end

zero(A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(zero, A.tensor), A.division)

function IU1(atype, dtype, dir, qn, dims; q::Vector{Int}=[0])
    L = length(dir)
    bkqn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    bkdims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(qn)))
        bkqni = [qn[j][i.I[j]] for j in 1:L]
        if sum(bkqni .* dir) in q
            bulkdims = [dims[j][i.I[j]] for j in 1:L]
            push!(bkqn, bkqni)
            push!(bkdims, bulkdims)
            push!(tensor, atype{dtype}(I, bulkdims...))
        end
    end
    U1Array(dir, qn, dims, bkqn, bkdims, tensor, 1)
end

size(A::U1Array) = Tuple(map(sum, A.dims))
size(A::U1Array, a) = size(A)[a]
getdir(A::U1Array) = A.dir
conj(A::U1Array) = U1Array(-A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(conj, A.tensor), A.division)
map(conj, A::U1Array) = conj(A)
norm(A::U1Array) = norm(A.tensor)

*(A::U1Array, B::Number) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, A.tensor * B, 1)
*(B::Number, A::U1Array) = A * B
/(A::U1Array, B::Number) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, A.tensor / B, 1)
broadcasted(*, A::U1Array, B::Number) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, A.tensor * B, 1)
broadcasted(*, B::Number, A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, A.tensor * B, 1)
broadcasted(/, A::U1Array, B::Number) = A / B

function +(A::U1Array, B::U1Array)
    if A.bkqn == B.bkqn
        U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, A.tensor + B.tensor, A.division)
    else
        bkqn = intersect(A.bkqn, B.bkqn)
        tensor = A.tensor[indexin(bkqn, A.bkqn)] + B.tensor[indexin(bkqn, B.bkqn)]
        extrabkqn = setdiff(A.bkqn, B.bkqn)            # setdiff result is dependent on order
        if length(extrabkqn) !== 0
            push!(bkqn, extrabkqn...)
            push!(tensor, A.tensor[indexin(extrabkqn, A.bkqn)]...)
        end
        extrabkqn = setdiff(B.bkqn, A.bkqn)
        if length(extrabkqn) !== 0
            push!(bkqn, extrabkqn...)
            push!(tensor, B.tensor[indexin(extrabkqn, B.bkqn)]...)
        end

        dims = map(x -> collect(size(x)), tensor)
        U1Array(A.dir, A.qn, dims, bkqn, A.bkdims, tensor, A.division)
    end
end

function -(A::U1Array, B::U1Array)
    if A.bkqn == B.bkqn
        U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, A.tensor - B.tensor, A.division)
    else
        # atype = typeof(A.tensor[1])
        # qn = intersect(A.qn, B.qn)
        # tensor = Vector{atype}(A.tensor[indexin(qn, A.qn)] - B.tensor[indexin(qn, B.qn)])
        # extraqn = setdiff(A.qn, B.qn)            # setdiff result is related to order
        # if length(extraqn) !== 0
        #     push!(qn, extraqn...)
        #     push!(tensor, A.tensor[indexin(extraqn, A.qn)]...)
        # end
        # extraqn = setdiff(B.qn, A.qn)
        # if length(extraqn) !== 0
        #     push!(qn, extraqn...)
        #     push!(tensor, -B.tensor[indexin(extraqn, B.qn)]...)
        # end

        # dims = map(x -> collect(size(x)), tensor)
        # U1Array(qn, A.dir, tensor, A.size, dims, A.division)
        exchangeind = indexin(A.bkqn, B.bkqn)
        U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, A.tensor - B.tensor[exchangeind], A.division)
    end
end

-(A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(-, A.tensor), A.division)

CuArray(A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(CuArray, A.tensor), A.division)
Array(A::U1Array) =  U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(Array, A.tensor), A.division)

function dot(A::U1Array, B::U1Array) 
    if A.bkqn == B.bkqn 
        dot(A.tensor, B.tensor)
    # elseif length(A.qn) > length(B.qn)
    #     # @warn "dot product of U1Array with different quantum numbers"
    #     # setdiff(B.qn, A.qn) !== [] && @show setdiff(B,A)
    #     exchangeind = indexin(B.qn, A.qn)
    #     dot(A.tensor[exchangeind], B.tensor)
    else
        # @warn "dot product of U1Array with different quantum numbers"
        commonbkqn = intersect(A.bkqn, B.bkqn)
        # @show commonbkqn A.bkqn B.bkqn
        # exchangeind = indexin(A.bkqn, B.bkqn)
        # @show indexin(commonbkqn, A.bkqn) indexin(commonbkqn, B.bkqn)
        commonbkqn == [] && return 0.0
        dot(A.tensor[indexin(commonbkqn, A.bkqn)], B.tensor[indexin(commonbkqn, B.bkqn)])
    end
end

function ≈(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    exchangeind = indexin(A.bkqn, B.bkqn)
    A.tensor ≈ B.tensor[exchangeind]
end

function ==(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    exchangeind = indexin(A.bkqn, B.bkqn)
    A.tensor == B.tensor[exchangeind]
end

# getindex(A::U1Array, index::CartesianIndex) = getindex(A, index.I...)
# function getindex(A::U1Array{T,N}, index::Int...) where {T,N}
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     qn = collect(map((index, bits) -> sum(minuseven(bitarray(index - 1, bits))), index, bits))
#     # sum(qn.*Adir) != 0 && return 0.0
#     ind = findfirst(x->x in [qn], A.qn)
#     ind === nothing && return 0.0
#     qlist = [U1selection(size(A, i)) for i = 1:N]
#     qrange = getqrange(size(A)...)
#     shift = getshift(qrange)
#     position = [sum(qlist[i][qn[i] + shift[i]][1:index[i]]) for i in 1:N]
#     CUDA.@allowscalar A.tensor[ind][position...]
# end

# setindex!(A::U1Array, x::Number, index::CartesianIndex) = setindex!(A, x, index.I...)
# function setindex!(A::U1Array{T,N}, x::Number, index::Int...) where {T,N}
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     qn = collect(map((index, bits) -> sum(minuseven(bitarray(index - 1, bits))), index, bits))
#     ind = findfirst(x->x in [qn], A.qn)
#     qlist = [U1selection(size(A, i)) for i = 1:N]
#     qrange = getqrange(size(A)...)
#     shift = getshift(qrange)
#     position = [sum(qlist[i][qn[i] + shift[i]][1:index[i]]) for i in 1:N]
#     CUDA.@allowscalar A.tensor[ind][position...] = x
# end

function U1selection(qn, dims)
    q = vcat([[qn[i] for _ in 1:dims[i]] for i in 1:length(qn)]...)
    [q .== i for i in qn]
end

function asArray(A::U1Array{T,N}) where {T,N}
    atype = _arraytype(A.tensor[1])
    tensor = zeros(T, size(A))
    Aqn = A.qn
    Adims = A.dims
    Abkqn = A.bkqn
    Atensor = A.tensor
    qlist = [U1selection(Aqn[i], Adims[i]) for i = 1:N]
    for i in 1:length(Abkqn)
        tensor[[qlist[j][indexin([Abkqn[i][j]], Aqn[j])...] for j = 1:N]...] = Array(Atensor[i])
    end
    atype(tensor)
end

function deletezerobulk(A::U1Array)
    nozeroind = norm.(A.tensor) .> 1e-30
    U1Array(A.dir, A.qn, A.dims, A.bkqn[nozeroind], A.bkdims[nozeroind], A.tensor[nozeroind], A.division)
end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function asU1Array(A::AbstractArray{T,N}, dir, Aqn, Adims; q::Vector{Int}=[0]) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [U1selection(Aqn[i], Adims[i]) for i = 1:N]
    Abkqn = getbkqn(dir, Aqn; q = q)
    tensor = [atype(Aarray[[qlist[j][indexin([Abkqn[i][j]], Aqn[j])...] for j = 1:N]...]) for i in 1:length(Abkqn)]
    bkdims = map(x -> collect(size(x)), tensor)
    deletezerobulk(U1Array(dir, Aqn, Adims, Abkqn, bkdims, tensor, 1))
end

# # only for OMEinsum binary permutedims before reshape
permutedims(A::U1Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::U1Array, perm)
    length(perm) == 0 && return copy(A)
    bkqn = map(x -> x[collect(perm)], A.bkqn)
    bkdims = map(x -> x[collect(perm)], A.bkdims)
    tensor = map(x -> permutedims(x, perm), A.tensor)
    U1Array(A.dir[collect(perm)], A.qn[collect(perm)], A.dims[collect(perm)], bkqn, bkdims, tensor, A.division)
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
        tensor = map((x, y) -> reshape(x, prod(y[1:div]), prod(y[div+1:end])), Atensor, A.bkdims)
        return U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, tensor, div)
    else
        tensor = map((x, y) -> reshape(x, y...), Atensor, A.bkdims)
        return U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, tensor, A.division)
    end
end

"""
    *(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,TB,NA,NB}

core code for U1Array product
"""
function *(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,TB,NA,NB}
    bkqn = Vector{Vector{Int}}()
    bkdims = Vector{Vector{Int}}()
    # qn_para = Vector{Vector{Vector{Int}}}()
    # dims_para = Vector{Vector{Vector{Int}}}()
    atype = _arraytype(B.tensor[1])
    T = TA == ComplexF64 || TB == ComplexF64 ? ComplexF64 : Float64
    tensor = Vector{atype{T}}()
    # tensor_para = Vector{Vector{atype{T}}}()
    divA, divB = A.division, B.division

    Adir = getdir(A)[divA+1:end]
    Bdir = getdir(B)[1:divB]
    Bdir != -Adir && throw(Base.error("U1Array product: out and in direction not match, expect: $(-Adir), got: $(Bdir)"))

    # oridims = [prod(A.size[1:divA]), prod(A.size[divA+1:end])], [prod(B.size[1:divB]), prod(B.size[divB+1:end])]
    # @show oridims
    for q in unique(map(bkqn -> sum(bkqn[divA+1:end] .* A.dir[divA+1:end]), A.bkqn))
        u1bulktimes!(bkqn, bkdims, tensor, A, B, q)
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
    bkqn == [[]] && return Array(tensor[1])[]
    ABdir = [A.dir[1:divA]..., B.dir[divB+1:end]...]
    ABqn = [A.qn[1:divA]..., B.qn[divB+1:end]...]
    ABdims = [A.dims[1:divA]..., B.dims[divB+1:end]...]
    U1Array(ABdir, ABqn, ABdims, bkqn, bkdims, tensor, divA)

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
    u1bulktimes!(bkqn, bkdims, tensor, A, B, q)

fill into different quantum number,  then dispatch to result tensor after product
"""
function u1bulktimes!(bkqn, bkdims, tensor, A, B, q)
    Abkqn, Atensor = A.bkqn, A.tensor
    Bbkqn, Btensor = B.bkqn, B.tensor
    Abkdims, Bbkdims = A.bkdims, B.bkdims
    divA, divB = A.division, B.division
    atype = _arraytype(Btensor[1])
    etype = eltype(Btensor[1])

    ind_A = findall(x->sum(x[divA+1:end] .* A.dir[divA+1:end]) == q, Abkqn)
    matrix_j = intersect(map(x->x[divA+1:end], Abkqn[ind_A]), map(x->x[1:divB], Bbkqn))
    ind_A = findall(x->x[divA+1:end] in matrix_j, Abkqn)
    matrix_i = unique(map(x->x[1:divA], Abkqn[ind_A]))
    ind_B = findall(x->x[1:divB] in matrix_j, Bbkqn)
    ind_B == [] && return
    matrix_k = unique(map(x->x[divB+1:end], Bbkqn[ind_B]))

    # @show Abkqn Bbkqn matrix_i matrix_j ind_A ind_B matrix_k
    index = [findfirst(x->x in [[i; j]], Abkqn) for i in matrix_i, j in matrix_j]
    # @show index
    if nothing in index
        indexcol = no_nothing_col(index)
        indexrow = no_nothing_row(index)
    else
        indexcol = index[:, 1]
        indexrow = index[1, :]
    end

    oribulkidims = map(ind -> Abkdims[ind][1:divA], indexcol)
    bulkidims = map(ind -> size(Atensor[ind], 1), indexcol)
    bulkjdims = map(ind -> size(Atensor[ind], 2), indexrow)
    # Amatrix = hvcat(ntuple(i->length(bulkjdims), length(bulkidims)), Atensor[index']...)
    Amatrix = atype <: Array ? zeros(etype, sum(bulkidims), sum(bulkjdims)) : CUDA.zeros(etype, sum(bulkidims), sum(bulkjdims))
    # @show size(Amatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        # println(sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), ", ", sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), " ", index[i, j])
        index[i, j] !== nothing && (Amatrix[sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])] .= Atensor[index[i, j]])
    end

    index = [findfirst(x->x in [[j; k]], Bbkqn) for j in matrix_j, k in matrix_k]
    oribulkkdims = []
    bulkkdims = Vector{Int}()
    indexrow = nothing in index ? no_nothing_row(index) : index[1, :]
    oribulkkdims = map(ind -> Bbkdims[ind][divB+1:end], indexrow)
    bulkkdims = map(ind -> size(Btensor[ind], 2), indexrow)
    # Bmatrix = hvcat(ntuple(i->length(bulkkdims), length(bulkjdims)), Btensor[index']...)
    Bmatrix = atype <: Array ? zeros(etype, sum(bulkjdims), sum(bulkkdims)) : CUDA.zeros(etype, sum(bulkjdims), sum(bulkkdims))
    # @show size(Bmatrix)
    for j in 1:length(matrix_j), k in 1:length(matrix_k)
        # println(sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), ", ", sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k]), " ", index[j, k])
        index[j, k] !== nothing && (Bmatrix[sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])] .= Btensor[index[j, k]])
    end
    
    # @show size(Amatrix), size(Bmatrix)
    C = Amatrix * Bmatrix

    for i in 1:length(matrix_i), k in 1:length(matrix_k)
        push!(bkqn, [matrix_i[i]; matrix_k[k]])
        push!(bkdims, [oribulkidims[i]; oribulkkdims[k]])
        idim, kdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])
        # println(idim, ", ", kdim)
        push!(tensor, C[idim, kdim])
    end
end


# # for OMEinsum contract to get number
# # vec(A::U1Array) = A

transpose(A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(transpose, A.tensor), 0)

function tr(A::U1Array{T,2}) where T
    bkqn = A.bkqn
    tensor = A.tensor
    s = 0.0
    @inbounds @simd for i in 1:length(bkqn)
        bkqn[i][1] == bkqn[i][2] && (s += tr(tensor[i]))
    end
    s
end

# function _compactify!(y, x::U1Array, indexer)
#     x = asArray(Array(x))
#     @inbounds @simd for ci in CartesianIndices(y)
#         y[ci] = x[subindex(indexer, ci.I)]
#     end
#     return y
# end

# broadcasted(*, A::U1Array, B::Base.RefValue) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)
# broadcasted(*, B::Base.RefValue, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)

# for ein"abab ->"(A)[]
function dtr(A::U1Array{T,N}) where {T,N}
    bkqn = A.bkqn
    tensor = A.tensor
    s = 0.0
    @inbounds @simd for i in 1:length(bkqn)
        bkqn[i][1] == bkqn[i][3] && bkqn[i][2] == bkqn[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end


# for Zygote compatibility
accum(A::U1Array, B::U1Array...) = +(A, B...)

# for KrylovKit compatibility
rmul!(A::U1Array, B::Number) = (map(x -> rmul!(x, B), A.tensor); A)

copy(A::U1Array) = U1Array([deepcopy(getfield(A, k)) for k ∈ fieldnames(U1Array)]...)
similar(A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(similar, A.tensor), A.division)
similar(A::U1Array, atype) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(x -> atype(similar(x)), A.tensor), A.division)
diag(A::U1Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))

function mul!(Y::U1Array, A::U1Array, B::Number)
    if Y.bkqn == A.bkqn
        map((Y, A) -> mul!(Y, A, B), Y.tensor, A.tensor)
    else
        # @warn "mul!(Y, A, B) : length(A.bkqn) !== length(B.bkqn)"
        exchangeind = indexin(A.bkqn, Y.bkqn)  # A.bkqn is a subset of Y.bkqn
        map((Y, A) -> mul!(Y, A, B), Y.tensor[exchangeind], A.tensor)
        # Y = A*B
    end
    # @show length.([Y.bkqn, Y.tensor])
    Y
end

function axpy!(α::Number, X::U1Array, Y::U1Array)
    if Y.bkqn == X.bkqn
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
        # length(X.qn) !== length(Y.qn) && throw(Base.error("$A and $B have different qn"))
        exchangeind = indexin(Y.bkqn, X.bkqn)
        map((x,y) -> axpy!(α, x, y), X.tensor[exchangeind], Y.tensor)
    end
    return Y
end

function axpby!(α::Number, x::U1Array, β::Number, y::U1Array)
    if x.bkqn == y.bkqn
        map((x,y) -> axpby!(α, x, β, y), x.tensor, y.tensor)
    else
        # length(x.qn) !== length(y.qn) && @warn "axpby!(x, β, y) is not implemented for x.qn != y.qn"
        # y = α * x + β * y
        exchangeind = indexin(y.bkqn, x.bkqn)
        map((x,y) -> axpby!(α, x, β, y), x.tensor[exchangeind], y.tensor)
    end

    # x.bkqn !== y.bkqn && @warn "axpby!(x, β, y) is not implemented for x.bkqn != y.bkqn"
    # map((x,y) -> axpby!(α, x, β, y), x.tensor, y.tensor)
    # return y
end

# # for leftorth and rightorth compatibility
Diagonal(A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(Diagonal, A.tensor), A.division)
sqrt(A::U1Array) = U1Array(A.dir, A.qn, A.dims, A.bkqn, A.bkdims, map(x->sqrt.(x), A.tensor), A.division)
broadcasted(sqrt, A::U1Array) = sqrt(A)
function lmul!(A::U1Array, B::U1Array)
    C = A*B
    for i = 1:length(B.bkqn)
        B.bkqn[i] = C.bkqn[i]
        B.bkdims[i] = C.bkdims[i]
        B.tensor[i] = C.tensor[i]
    end
    return B
end

# only for order-three tensor's qr and lq
function qrpos!(A::U1Array{T,N}) where {T,N}
    Qbkqn = Vector{Vector{Int}}()
    Rbkqn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Qtensor = Vector{atype{T}}()
    Rtensor = Vector{atype{T}}()

    for q in unique(map(x->sum(x[A.division+1:end] .* A.dir[A.division+1:end]), A.bkqn))
        u1bulkQR!(Qbkqn, Qtensor, Rbkqn, Rtensor, A, q)
    end
    exchangeind = indexin(Qbkqn, A.bkqn)
    U1Array(A.dir, A.qn, A.dims, Qbkqn, A.bkdims[exchangeind], Qtensor, A.division), U1Array([-A.dir[end], A.dir[end]], [A.qn[end], A.qn[end]], [A.dims[end], A.dims[end]], Rbkqn, map(x -> [size(x)...], Rtensor), Rtensor, 1)
end

function u1bulkQR!(Qbkqn, Qtensor, Rbkqn, Rtensor, A, q)
    Atensor = A.tensor
    Abkqn = A.bkqn
    Adiv = A.division

    ind_A = findall(x->sum(x[Adiv+1:end] .* A.dir[Adiv+1:end]) == q, Abkqn)
    matrix_j = unique(map(x->x[Adiv+1:end], Abkqn[ind_A]))
    matrix_i = unique(map(x->x[1:Adiv], Abkqn[ind_A]))

    ind = [findfirst(x->x in [[i; matrix_j[1]]], Abkqn) for i in matrix_i]
    Amatrix = vcat(Atensor[ind]...)
    bulkidims = [size(Atensor[i],1) for i in ind]
    bulkjdims = [size(Amatrix, 2)]

    Q, R = qrpos!(Amatrix)
    for i in 1:length(matrix_i)
        push!(Qbkqn, [matrix_i[i]; matrix_j[1]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), 1:sum(bulkjdims[1])
        push!(Qtensor, Q[idim, jdim])
    end
    
    push!(Rbkqn, [matrix_j[1]; matrix_j[1]])
    idim, jdim = 1:sum(bulkjdims[1]), 1:sum(bulkjdims[1])
    push!(Rtensor, R[idim, jdim])
end

function lqpos!(A::U1Array{T,N}) where {T,N}
    Lbkqn = Vector{Vector{Int}}()
    Qbkqn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Ltensor = Vector{atype{T}}()
    Qtensor = Vector{atype{T}}()

    for q in unique(map(x->x[1] * A.dir[1], A.bkqn))
        u1bulkLQ!(Lbkqn, Ltensor, Qbkqn, Qtensor, A, q)
    end
    exchangeind = indexin(Qbkqn, A.bkqn)
    U1Array([A.dir[1], -A.dir[1]], [A.qn[1], A.qn[1]], [A.dims[1], A.dims[1]], Lbkqn, map(x -> [size(x)...], Ltensor), Ltensor, 1), U1Array(A.dir, A.qn, A.dims, Qbkqn, A.bkdims[exchangeind], Qtensor, A.division)
end

function u1bulkLQ!(Lbkqn, Ltensor, Qbkqn, Qtensor, A, q)
    Atensor = A.tensor
    Abkqn = A.bkqn
    Adiv = A.division

    ind_A = findall(x->x[1] * A.dir[1] == q, Abkqn)
    matrix_j = unique(map(x->x[Adiv+1:end], Abkqn[ind_A]))
    matrix_i = unique(map(x->x[1], Abkqn[ind_A]))

    ind = [findfirst(x->x in [[matrix_i[1]; j]], Abkqn) for j in matrix_j]
    Amatrix = hcat(Atensor[ind]...)
    bulkidims = [size(Amatrix, 1)]
    bulkjdims = [size(Atensor[i], 2) for i in ind]
    
    L, Q = lqpos!(Amatrix)

    push!(Lbkqn, [matrix_i[1]; matrix_i[1]])
    idim, jdim = 1:sum(bulkidims[1]), 1:sum(bulkidims[1])
    push!(Ltensor, L[idim, jdim])
    for j in 1:length(matrix_j)
        push!(Qbkqn, [matrix_i[1]; matrix_j[j]])
        idim, jdim = 1:sum(bulkidims[1]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
end

# # for ' in ACCtoALAR of VUMPS
function adjoint(A::U1Array{T,N}) where {T,N}
    div = A.division 
    bkqn = map(x->x[[div+1:end;1:div]], A.bkqn)
    bkdims = map(x -> x[[div+1:end;1:div]], A.bkdims)
    tensor = map(adjoint, A.tensor)
    U1Array(-A.dir[[div+1:end;1:div]], A.qn[[div+1:end;1:div]], A.dims[[div+1:end;1:div]], bkqn, bkdims, tensor, N - div)
end

# only for U1 square Matrix
function sysvd!(A::U1Array{T,N}) where {T,N}
    # Atensor = asArray(A)
    # Utensor, Stensor, Vtensor = sysvd!(Atensor)
    # dir = getdir(A)
    # U = asU1Array(Utensor; dir = dir, q=collect(-2:2))
    # S = asU1Array(Diagonal(Stensor); dir = dir, q=[0])
    # V = asU1Array(Vtensor; dir = dir, q=collect(-2:2))
    # return U, S, V
    Abkqn = A.bkqn
    Abkdims = A.bkdims
    tensor = A.tensor
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
    bkdimsm = map(x->min(x...), Abkdims)
    bkdims1 = map((x, y) -> [x[1], y], Abkdims, bkdimsm)
    bkdims2 = map((x, y) -> [y, x[2]], Abkdims, bkdimsm)
    U1Array(A.dir, A.qn, A.dims, Abkqn, bkdims1, Utensor, div), U1Array(A.dir, A.qn, A.dims, Abkqn, [[bkdimsm[i], bkdimsm[i]] for i in 1:length(Abkqn)], Stensor, div), U1Array(A.dir, A.qn, A.dims, Abkqn, bkdims2, Vtensor, div)
end

"""
    U1reshape(A::U1Array{T, N}, a::Int...) where {T, N}

U1reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`, and the high-oreder U1tensor is from randU1 or zerosU1 function.
"""
U1reshape(A::U1Array, s::Tuple{Vararg{Int}}; kwarg...) = U1reshape(A, s...; kwarg...)
function U1reshape(A::U1Array{T, N}, s::Int...; reinfo = nothing) where {T, N}
    atype = _arraytype(A.tensor[1])
    if N > length(s)
        cA = zerosU1(Array, ComplexF64, A.dir, A.qn, A.dims)
        bkqndiff = setdiff(cA.bkqn, A.bkqn)
        supind = indexin(bkqndiff, cA.bkqn)
        Abkqn = [A.bkqn; cA.bkqn[supind]]
        Atensor = [A.tensor; cA.tensor[supind]]
        exchangeind = indexin(cA.bkqn, Abkqn)
        Abkqn = cA.bkqn
        Abkdims = cA.bkdims
        Atensor = Atensor[exchangeind]
        div = division(s, size(A))
        rebkqn = [[sum(q[d] .* A.dir[d]) for d in div] for q in Abkqn]
        rebkdims = [[prod(bkdims[d]) for d in div] for bkdims in Abkdims]
        retensor = [reshape(t, s...) for (t, s) in zip(map(Array, Atensor), rebkdims)]
        urebkqn = unique(rebkqn)
        retensors = Vector{atype{T}}()
        indqn = cA.qn
        inddims = cA.dims
        choosesilces = [[] for _ in 1:length(urebkqn)]
        chooseinds = [[] for _ in 1:length(urebkqn)]
        for i in 1:length(urebkqn)
            q = urebkqn[i]
            bulkind = findall(x->x in [q], rebkqn)
            oribkqn = Abkqn[bulkind]
        
            indbkqnfrom = [unique(map(x->x[div], oribkqn)) for div in div]
            rebulkdims = [[prod(map((x,y,s)->x[indexin(y, s)...], inddims[div[i]], indbkqnfrom, indqn[div[i]])) for indbkqnfrom in indbkqnfrom[i]] for i in 1:length(indbkqnfrom)]
            # @show indbkqnfrom
            # indbkqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
            # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
            silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
            tensor = atype(zeros(T, map(sum, rebulkdims)...))
            for j in 1:length(bulkind)
                chooseind = [indexin([oribkqn[j][div[i]]], indbkqnfrom[i]) for i in 1:length(div)]
                choosesilce = map((s,i)->s[i...], silce, chooseind)
                tensor[choosesilce...] = retensor[bulkind[j]]
                push!(choosesilces[i], choosesilce)
                push!(chooseinds[i], bulkind[j])
            end
            push!(retensors, tensor)
        end
        bkdims = map(x -> collect(size(x)), retensors)
        dir = [A.dir[d][end] for d in div]     # last dir of reshape
        deletezerobulk(U1Array(dir,[[s] for s in s], [[s] for s in s], map(bkqn->bkqn .* dir, urebkqn), bkdims, map(atype, retensors), 1)), (choosesilces, chooseinds, A.dir, cA.qn, cA.dims, Abkqn, Abkdims)
    else
        choosesilces, chooseinds, redir, reqn, redims, rebkqn, rebkdims = reinfo
        retensors = Array{Array,1}(undef, sum(length.(chooseinds)))
        div = division(size(A), s)
        urebkqn = unique([[sum(q[d] .* redir[d]) for d in div] for q in rebkqn])
        exchangeind = indexin(urebkqn, map(bkqn->bkqn .* A.dir, A.bkqn))
        # @show urebkqn A.bkqn exchangeind
        Atensor = A.tensor[exchangeind]
        for i in 1:length(choosesilces)
            for j in 1:length(choosesilces[i])
                retensors[chooseinds[i][j]] = reshape(Array(Atensor[i][choosesilces[i][j]...]), rebkdims[chooseinds[i][j]]...)
            end
        end
        bkdims = map(x -> collect(size(x)), retensors)
        deletezerobulk(U1Array(redir, reqn, redims, rebkqn, rebkdims, map(atype, retensors), 1)), (choosesilces, chooseinds, rebkqn, redir, redims)
    end
end

function U1reshapeinfo(s, sizeA, dir)
    length(sizeA) < length(s) && throw(Base.error("$sizeA must be longer than $s"))
    div = division(s, sizeA)
    A = zerosU1(Array, Float64, sizeA...; dir = dir)
    rebkqn = [[sum(p[d] .* dir[d]) for d in div] for p in A.bkqn]
    urebkqn = unique(rebkqn)
    inddims = u1bulkdims(sizeA...)   # only correct for bits division U1Array
    choosesilces = [[] for _ in 1:length(urebkqn)]
    chooseinds = [[] for _ in 1:length(urebkqn)]
    Abkbkqn = A.bkqn
    qrange = getqrange(size(A)...)
    shift = getshift(qrange)
    for i in 1:length(urebkqn)
        q = urebkqn[i]
        bulkind = findall(x->x in [q], rebkqn)
        oribkqn = Abkbkqn[bulkind]
    
        indbkqnfrom = [unique(map(x->x[div], oribkqn)) for div in div]
        rebulkdims = [[prod(map((x,y,s)->x[y+s], inddims[div[i]], indbkqnfrom, shift[div[i]])) for indbkqnfrom in indbkqnfrom[i]] for i in 1:length(indbkqnfrom)]
        # @show indbkqnfrom
        # indbkqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
        # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
        silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
        for j in 1:length(bulkind)
            chooseind = [indexin([oribkqn[j][div[i]]], indbkqnfrom[i]) for i in 1:length(div)]
            choosesilce = map((s,i)->s[i...], silce, chooseind)
            push!(choosesilces[i], choosesilce)
            push!(chooseinds[i], bulkind[j])
        end
    end
    choosesilces, chooseinds, Abkqn, dir, A.dims
end