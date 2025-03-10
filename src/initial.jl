"""
Initalize a boundary MPS for the transfer operator `O` by specifying an array of virtual
spaces consistent with the unit cell.

````

   l ←------- r
        / \
       /   \
      t     d
````
"""
function initial_A(M::bulk, χ::VectorSpace)
    T = eltype(M[1])
    A = rand(T, [(D = space(M, 4)'; χ*D ← χ)  for M in M.data], M.pattern)
    return _arraytype(M[1])(A)
end

function initial_A(M::ipeps, χ::VectorSpace)
    T = eltype(M[1])
    A = rand(T, [(D = space(M, 4)'; χ*D*D' ← χ)  for M in M.data], M.pattern)
    return _arraytype(M[1])(A)
end

# function initial_A(M::doubleipeps, χ::VectorSpace)
#     T = eltype(M[1])
#     A = rand(T, [(D = space(M, 8)'; χ*D*D' ← χ)  for M in M.data], M.pattern)
#     return _arraytype(M[1])(A)
# end

"""
````
   l ←------- r
````
"""
function initial_C(A::StructArray)
    T = eltype(A[1])
    C = StructArray([(χ = space(A, 1); isomorphism(T, χ, χ)) for A in A.data], A.pattern)
    return C
end

function initial_FL(AL::leg3, M::bulk)
    T = eltype(M[1])
    FL = rand(T, [(D = space(M, 1)';
                   χ = space(AL, 1);
                   χ*D ← χ) for (M, AL) in zip(M.data, AL.data)
                 ], 
              M.pattern
    )
    return FL
end

function initial_FL(AL::leg4, M::ipeps)
    T = eltype(M[1])
    FL = rand(T, [(D = space(M,  1)';
                   χ = space(AL, 1);
                   χ*D*D' ← χ) for (M, AL) in zip(M.data, AL.data)
                 ], 
              M.pattern
    )
    return FL
end

# function initial_FL(AL::leg4, M::doubleipeps)
#     T = eltype(M[1])
#     FL = rand(T, [(D = space(M,  1)';
#                    χ = space(AL, 1);
#                    χ*D*D' ← χ) for (M, AL) in zip(M.data, AL.data)
#                  ], 
#               M.pattern
#     )
#     return FL
# end

function initial_FR(AR::leg3, M::bulk)
    T = eltype(M[1])
    FR = rand(T, [(D = space(M,  3)';
                   χ = space(AR, 3)';
                   χ*D ← χ) for (M, AR) in zip(M.data, AR.data)
                 ], 
              M.pattern
    )
    return FR
end

function initial_FR(AR::leg4, M::ipeps)
    T = eltype(M[1])
    FR = rand(T, [(D = space(M,  3)';
                   χ = space(AR, 4)';
                   χ*D*D' ← χ) for (M, AR) in zip(M.data, AR.data)
                 ], 
              M.pattern
    )
    return FR
end
