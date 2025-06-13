const leg3 = Union{<:AbstractTensorMap{ComplexF64,S,2,1} where S, StructArray{<:Vector{<:AbstractTensorMap{ComplexF64,S,2,1}} where S},
                   <:AbstractTensorMap{ComplexF64,S,1,2} where S, StructArray{<:Vector{<:AbstractTensorMap{ComplexF64,S,1,2}} where S}}
const leg4 = Union{<:AbstractTensorMap{ComplexF64,S,3,1} where S, StructArray{<:Vector{<:AbstractTensorMap{ComplexF64,S,3,1}} where S},
                   <:AbstractTensorMap{ComplexF64,S,1,3} where S, StructArray{<:Vector{<:AbstractTensorMap{ComplexF64,S,1,3}} where S}}
const bulk  = Union{<:AbstractTensorMap{ComplexF64,S,2,2} where S, StructArray{<:Vector{<:AbstractTensorMap{ComplexF64,S,2,2}} where S}}
const ipeps = Union{<:AbstractTensorMap{ComplexF64,S,4,1} where S, StructArray{<:Vector{<:AbstractTensorMap{ComplexF64,S,4,1}} where S}}
# const doubleipeps = Union{<:AbstractTensorMap{ComplexF64,S,4,4} where S, StructArray{<:Vector{<:AbstractTensorMap{ComplexF64,S,4,4}} where S}}

function _to_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return permute(t, ((I1..., reverse(Base.tail(I2))...), (I2[1],)))
end

function _to_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return permute(t, ((I1[1],), (I2..., reverse(Base.tail(I1))...)))
end

permute_fronttail(t::leg3) = permute(t, ((3,2),   (1,)))
permute_fronttail(t::leg4) = permute(t, ((4,2,3), (1,)))
permute_fronttail(t::AbstractZero) = t

orth_for_ad(v) = v
function simple_eig(f, v; max_iter=5, ifvalue=false)
    λ = 0.0
    # Zygote.@ignore begin
        for _ in 1:max_iter
            v = f(v)
            λ′ = norm(v)
            v /= λ′
            abs(λ′ - λ) < 1e-8 && break
            λ = λ′
        end
    # end
    # for _ in 1:max_iter
    #     v = f(v)
    #     v /= norm(v)
    # end

    v = orth_for_ad(v)
    if ifvalue
        CUDA.@allowscalar λ = f(v)[1] ./ v[1]
    end
    return λ, v
end

# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, x...; kwargs...) = f(x...; kwargs...) 
Zygote.@adjoint checkpoint(f, x...; kwargs...) = f(x...; kwargs...), ȳ -> Zygote._pullback(f, x...)[2](ȳ)

to_CuArray(x) = map(CuArray, x)
to_Array(x) = map(Array, x)

# make two TensorMap's have the same spaces, by force if necessary
# this is definitely not what you would want to do, but it circumvents having to think
# about what hermiticity means at the level of transfer operators, which is something
function _fit_spaces(
    y::AbstractTensorMap{T,S,N₁,N₂}, x::AbstractTensorMap{T,S,N₁,N₂}
) where {T,S<:IndexSpace,N₁,N₂}
    for i in 1:(N₁ + N₂)
        if space(x, i).dual ≠ space(y, i).dual
            f = unitary(space(y, i)' ← space(y, i))
            y = permute(
                ncon([f, y], [[-i, 1], [-(1:(i - 1))..., 1, -((i + 1):(N₁ + N₂))...]]),
                (Tuple(1:N₁), Tuple((N₁ + 1):(N₁ + N₂))),
            )
        end
    end
    return y
end