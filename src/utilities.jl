#helper functions to handle array types
_mattype(::Array) = Matrix
_mattype(::CuArray) = CuMatrix
_mattype(::ROCArray) = ROCMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::ROCArray) = ROCArray
_arraytype(S::StructArray) = _arraytype(S.data[1])

const leg3 = Union{<:AbstractArray{T, 3}, StructArray{<:Vector{<:AbstractArray{T, 3}}}} where T
const leg4 = Union{<:AbstractArray{T, 4}, StructArray{<:Vector{<:AbstractArray{T, 4}}}} where T
const leg5 = Union{<:AbstractArray{T, 5}, StructArray{<:Vector{<:AbstractArray{T, 5}}}} where T
const leg8 = Union{<:AbstractArray{T, 8}, StructArray{<:Vector{<:AbstractArray{T, 8}}}} where T

function _to_front(t)
    χ = size(t)[end]
    return reshape(t, χ, Int(prod(size(t))/χ))
end

function _to_tail(t)
    χ = size(t, 1)
    return reshape(t, Int(prod(size(t))/χ), χ)
end

permute_fronttail(t::leg3) = permutedims(t, (3,2,1))
permute_fronttail(t::leg4) = permutedims(t, (4,2,3,1))
permute_fronttail(t::InnerProductVec) = RealVec(permute_fronttail(t.vec))
permute_fronttail(t::AbstractZero) = t

orth_for_ad(v) = v
function simple_eig(f, v; max_iter=10, ifvalue=false)
    λ = 0.0
    # Zygote.@ignore begin # this is not correct when VUMPS does not converge
    #     for _ in 1:max_iter
    #         v = f(v)
    #         λ′ = norm(v)
    #         v /= λ′
    #         abs(λ′ - λ) < 1e-8 && break
    #         λ = λ′
    #     end
    # end
    for _ in 1:max_iter
        v = f(v)
        v /= norm(v)
    end

    v = orth_for_ad(v)
    if ifvalue
        CUDA.@allowscalar λ = f(v)[1] ./ v[1]
    end
    return λ, v
end

function mcform(M)
    aM = Array(M)
    x = ein"ijil->jl"(aM)
    _, vh = Zygote.@ignore eigen(x)
    aM = ein"aj,(ijkl,lb)->iakb"(inv(vh),aM,vh)
    y = ein"ijkj->ik"(aM)
    _, vv = Zygote.@ignore eigen(y)
    aM = ein"(ai,ijkl),kb->ajbl"(inv(vv),aM,vv)
    aM = typeof(M)(aM)
    return vh, vv, aM
end    

# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, x...; kwargs...) = f(x...; kwargs...) 
Zygote.@adjoint checkpoint(f, x...; kwargs...) = f(x...; kwargs...), ȳ -> Zygote._pullback(f, x...)[2](ȳ)

function save_rt(folder, rt)
    p = joinpath(folder, "VUMPS_rt.jld2")
    if rt.AL[1] isa CuArray
        data = []
        for field in fieldnames(typeof(rt))
            A = getfield(rt, field)
            push!(data, Array(A))
        end
        rt_save = VUMPSRuntime(data...)
        println("save a CuArray rt in $p")
        save(p, "rt", rt_save)
    else
        println("save a Array rt in $p")
        save(p, "rt", rt)
    end
end

function load_rt(folder, atype)
    p = joinpath(folder, "VUMPS_rt.jld2")
    rt = load(p, "rt")
    if atype == CuArray
        data = []
        for field in fieldnames(typeof(rt))
            A = getfield(rt, field)
            push!(data, CuArray(A))
        end
        println("load a CuArray rt in $p")
        rt = VUMPSRuntime(data...)
    else
        println("load a Array rt in $p")
    end
    return rt
end

set_device_id!(::Type{ROCArray}, i::Int) = AMDGPU.device_id!(i)
set_device_id!(::Type{CuArray}, i::Int) = CUDA.device!(i-1)
set_device_id!(::Type{Array}, i::Int) = nothing

get_device(::Type{ROCArray}) = AMDGPU.device()
get_device(::Type{CuArray}) = CUDA.device()
get_device(::Type{Array}) = "CPU thread $(threadid())"

get_device_id(::Array) = 1
get_device_id(x::ROCArray) = Int(AMDGPU.device(x).device_id)
get_device_id(x::CuArray) = Int(CUDA.device(x).handle + 1)
get_device_id(S::StructArray) = get_device_id(S[1]) 

get_device_id(::Type{Array}) = threadid()
get_device_id(::Type{ROCArray}) = AMDGPU.device_id()
get_device_id(::Type{CuArray}) = CUDA.device().handle

function atype_device!(atype, x, i::Int)
    set_device_id!(atype, i)
    return atype(x)
end

function ROCArray(x::NamedTuple)
    x.data .= map(ROCArray, x.data)
    return x
end

Array(x::NamedTuple) = x
