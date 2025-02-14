const leg3 = Union{<:AbstractArray{ComplexF64, 3}, StructArray{<:Vector{<:AbstractArray{ComplexF64, 3}}}}
const leg4 = Union{<:AbstractArray{ComplexF64, 4}, StructArray{<:Vector{<:AbstractArray{ComplexF64, 4}}}}
const leg5 = Union{<:AbstractArray{ComplexF64, 5}, StructArray{<:Vector{<:AbstractArray{ComplexF64, 5}}}}
const leg8 = Union{<:AbstractArray{ComplexF64, 8}, StructArray{<:Vector{<:AbstractArray{ComplexF64, 8}}}}

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
function simple_eig(f, v; max_iter=20, ifvalue=true)
    λ = 0.0
    Zygote.@ignore begin
        for _ in 1:max_iter
            v = f(v)
            λ′ = norm(v)
            v /= λ′
            abs(λ′ - λ) < 1e-8 && break
            λ = λ′
        end
    end
    for _ in 1:5
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
Zygote.@adjoint checkpoint(f, x...; kwargs...) = f(x...; kwargs...), ȳ -> Zygote._pullback(f, x...; kwargs...)[2](ȳ)

to_CuArray(x) = map(CuArray, x)
to_Array(x) = map(Array, x)