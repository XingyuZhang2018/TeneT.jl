function cellones(A)
    χ = size(A[1], 1)
    return ISA(A, [(χ,χ) for _ = 1:length(A.data)])
end

function initial_A(M::leg4, χ::Int)
    return randSA(M, [(D = size(m, 4); (χ, D, χ)) for m in M.data])
end

function initial_A(M::leg5, χ::Int)
    return randSA(M, [(D = size(m, 4); (χ, D, D, χ)) for m in M.data])
end

function initial_A(M::leg8, χ::Int)
    return randSA(M, [(D = size(m, 7); (χ, D, D, χ)) for m in M.data])
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
