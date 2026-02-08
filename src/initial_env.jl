function cellones(A)
    χ = size(A[1], 1)
    return ISA(A, [(χ,χ) for _ = 1:length(A.data)])
end

function initial_Au(M::leg4, χ::Int)
    return randSA(M, [(D = size(m, 4); (χ, D, χ)) for m in M.data])
    data = []
    for m in M.data
        m_new = zeros(eltype(m), (χ, size(m,4), χ))
        D1, D3 = size(m)[[1,3]]
        m_new[1:D1, :, 1:D3] = ein"abcd -> abc"(m)
        push!(data, normalize(m_new))
    end
    return StructArray(data, M.pattern)
end

function initial_Au(M::leg5, χ::Int)
    return randSA(M, [(D = size(m, 4); (χ, D, D, χ)) for m in M.data])
end

function initial_Au(M::leg8, χ::Int)
    return randSA(M, [(D = size(m, 7); (χ, D, D, χ)) for m in M.data])
end

function initial_Ad(M::leg4, χ::Int)
    return randSA(M, [(D = size(m, 2); (χ, D, χ)) for m in M.data])
    data = []
    for m in M.data
        m_new = zeros(eltype(m), (χ, size(m,2), χ))
        D1, D3 = size(m)[[1,3]]
        # m_new[1:D1, :, 1:D3] = permutedims(sum(m, dims=2), (1,3,2))
        m_new[1:D1, :, 1:D3] = ein"abcd->adc"(m)
        push!(data, normalize(m_new))
    end
    return StructArray(data, M.pattern)
end

function initial_Ad(M::leg5, χ::Int)
    return randSA(M, [(D = size(m, 2); (χ, D, D, χ)) for m in M.data])
end

function initial_Ad(M::leg8, χ::Int)
    return randSA(M, [(D = size(m, 2); (χ, D, D, χ)) for m in M.data])
end


function FLint(AL, M::leg4)
    χ = size(AL[1], 1)
    return randSA(M, [(D = size(m, 1); (χ, D, χ)) for m in M.data])
end

function FLint(AL, M::leg5)
    χ = size(AL[1], 1)
    return randSA(M, [(D = size(m, 1); (χ, D, D, χ)) for m in M.data])
end

function FLint(AL, M::leg8)
    χ = size(AL[1], 1)
    return randSA(M, [(D = size(m, 1); (χ, D, D, χ)) for m in M.data])
end

function FRint(AR, M::leg4)
    χ = size(AR[1], 1)  
    return randSA(M, [(D = size(m, 3); (χ, D, χ)) for m in M.data])
end

function FRint(AR, M::leg5)
    χ = size(AR[1], 1)
    return randSA(M, [(D = size(m, 3); (χ, D, D, χ)) for m in M.data])
end

function FRint(AR, M::leg8)
    χ = size(AR[1], 1)
    return randSA(M, [(D = size(m, 5); (χ, D, D, χ)) for m in M.data])
end
