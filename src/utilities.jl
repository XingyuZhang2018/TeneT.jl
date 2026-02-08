#helper functions to handle array types
_mattype(::Array) = Matrix
_mattype(::CuArray) = CuMatrix
_mattype(::ROCArray) = ROCMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::ROCArray) = ROCArray
_arraytype(S::StructArray) = _arraytype(S.data[1])

const leg2 = Union{<:AbstractArray{T, 2}, StructArray{<:Vector{<:AbstractArray{T, 2}}}} where T
const leg3 = Union{<:AbstractArray{T, 3}, StructArray{<:Vector{<:AbstractArray{T, 3}}}} where T
const leg4 = Union{<:AbstractArray{T, 4}, StructArray{<:Vector{<:AbstractArray{T, 4}}}} where T
const leg5 = Union{<:AbstractArray{T, 5}, StructArray{<:Vector{<:AbstractArray{T, 5}}}} where T
const leg8 = Union{<:AbstractArray{T, 8}, StructArray{<:Vector{<:AbstractArray{T, 8}}}} where T

function _to_tail(t)
    χ = size(t)[end]
    return reshape(t, χ, Int(prod(size(t))/χ))
end

function _to_front(t)
    χ = size(t, 1)
    return reshape(t, Int(prod(size(t))/χ), χ)
end

permute_fronttail(t::leg3) = permutedims(t, (3,2,1))
permute_fronttail(t::leg4) = permutedims(t, (4,2,3,1))
permute_fronttail(t::InnerProductVec) = RealVec(permute_fronttail(t.vec))
permute_fronttail(t::AbstractZero) = t

orth_for_ad(v) = v
function simple_eig(f, v; power_iter, ifvalue=false)
    λ = 0.0
    # Zygote.@ignore begin # this is not correct when VUMPS does not converge
        for i in 1:power_iter
            v = f(v)
            λ′ = norm(v)
            v /= λ′
            # if abs(λ′ - λ) < 1e-12
            #     # @show i abs(λ′ - λ)
            #     break
            # end
            λ = λ′
        end
    # end
    # for _ in 1:power_iter
    #     v = f(v)
    #     v /= norm(v)
    # end

    v = orth_for_ad(v)
    if ifvalue
        # CUDA.@allowscalar λ = f(v)[1] ./ v[1]
        λ = dot(v, f(v))
    end
    return λ, v
end

# function mcform(M)
#     aM = Array(M)
#     x = ein"ijil->jl"(aM)
#     _, vh = Zygote.@ignore eigen(x)
#     aM = ein"aj,(ijkl,lb)->iakb"(inv(vh),aM,vh)
#     y = ein"ijkj->ik"(aM)
#     _, vv = Zygote.@ignore eigen(y)
#     aM = ein"(ai,ijkl),kb->ajbl"(inv(vv),aM,vv)
#     aM = typeof(M)(aM)
#     return vh, vv, aM
# end    

# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, args...; kwargs...) = f(args...; kwargs...) 
Zygote.@adjoint checkpoint(f, args...; kwargs...) = f(args...; kwargs...), ȳ -> Zygote._pullback((args...) -> f(args...; kwargs...), args...)[2](ȳ)

function save_rt(folder, rt; file::String="VUMPS_rt_env.jld2")
    p = joinpath(folder, file)
    rt_save = Array(rt)
    @info "save a VUMPS runtime environment to $p"
    save(p, "rt", rt_save)
end

function load_rt(folder, atype, ifparallelupdown; file::String="VUMPS_rt_env.jld2")
    p = joinpath(folder, file)
    rt = load(p, "rt")
    if ifparallelupdown
        rtup, rtdown = rt
        @sync begin
            @async begin
                TeneT.set_device_id!(atype, 1)
                rtup = atype(rtup)
            end
            @async begin
                TeneT.set_device_id!(atype, 2)
                rtdown = atype(rtdown)
            end
        end
        rt = (rtup, rtdown)
    else
        rt = atype(rt)
    end
    @info "load a VUMPS runtime environment from $p"
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
get_device_id(::Type{CuArray}) = Int(CUDA.device().handle) + 1

function atype_device!(atype, x, i::Int)
    set_device_id!(atype, i)
    return atype(x)
end

function ROCArray(x::NamedTuple)
    x.data .= map(ROCArray, x.data)
    return x
end

function CuArray(x::NamedTuple)
    x.data .= map(CuArray, x.data)
    return x
end

Array(x::NamedTuple) = x

function reclaim(x::AbstractArray) 
    if x isa CuArray && CUDA.available_memory() / CUDA.total_memory() < 0.1
        GC.gc(true)
        CUDA.reclaim()
    elseif x isa ROCArray
        GC.gc(true)
        AMDGPU.HIP.reclaim()
    end
end

for_gc(x) = x
function ChainRulesCore.rrule(::typeof(for_gc), x)
    function back(dx)
        reclaim(x[1])
        return NoTangent, dx
    end
    return x, back
end

function Z2_transform_matrix_χ(χ)
    sqrtχ = ceil(Int,sqrt(χ))
    U = zeros(Float64, sqrtχ, sqrtχ, sqrtχ, sqrtχ)
    c = 1/sqrt(2)

    @inbounds for i in 1:sqrtχ
        U[i,i,i,i] = 1
        for j in i+1:sqrtχ
            U[i,j,i,j] = c
            U[i,j,j,i] = c
            U[j,i,i,j] = c
            U[j,i,j,i] = -c
        end
    end

    return reshape(U, sqrtχ^2,sqrtχ^2)[1:χ, 1:χ]
end

function Z2_transform_matrix_D(D)
    U = zeros(Float64, D, D, D, D)
    c = 1/sqrt(2)

    @inbounds for i in 1:D
        U[i,i,i,i] = 1
        for j in i+1:D
            U[i,j,i,j] = c
            U[i,j,j,i] = c
            U[j,i,i,j] = c
            U[j,i,j,i] = -c
        end
    end

    return U
end

function convert_bilayer_Z2(M::AbstractArray{Float64, 2})
    χ = size(M, 1)
    U = _arraytype(M)(Z2_transform_matrix_χ(χ))
    @tensor M[1,2] = M[3,4] * U[3,1] * U[4,2]
    return M
end

function convert_bilayer_Z2(M::AbstractArray{Float64, 4})
    χ, D = size(M)[[1,2]]
    Uχ = _arraytype(M)(Z2_transform_matrix_χ(χ))
    UD = _arraytype(M)(Z2_transform_matrix_D(D))
    @tensor M[1,2,3,4] = M[5,6,7,8] * Uχ[5,1] * Uχ[8,4] * UD[6,7,2,3]
    return M
end

function convert_bilayer_Z2(M::AbstractArray{ComplexF64, N}) where {N}
    M_real = convert_bilayer_Z2(real(M))
    M_imag = convert_bilayer_Z2(imag(M))
    return M_real + 1im * M_imag
end

to_Z2(T::StructArray) = StructArray(to_Z2.(T.data), T.pattern)

function to_Z2(T::AbstractArray{Type, 2}) where {Type}
    χ = size(T, 1)
    sqrtχ = Int(sqrt(χ))
    T = reshape(T, sqrtχ, sqrtχ, sqrtχ, sqrtχ)
    T += conj(permutedims(T, (2,1,4,3)))
    T = reshape(T, χ, χ) / 2
    return T
end   

function to_Z2(T::AbstractArray{Type, 4}) where {Type}
    χ, D = size(T)[[1,2]]
    sqrtχ = Int(sqrt(χ))
    T = reshape(T, sqrtχ, sqrtχ, D, D, sqrtχ, sqrtχ)
    T += conj(permutedims(T, (2,1,4,3,6,5)))
    T = reshape(T, χ, D, D, χ) / 2
    return T
end   