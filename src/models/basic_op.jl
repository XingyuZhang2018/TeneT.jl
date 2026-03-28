# Spin operator constructors and generic Hamiltonian interface

function const_Sx(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sx = zeros(Float64, dims, dims)
    for j in 1:dims, i in 1:dims
        if abs(i-j) == 1
            Sx[i,j] = 1/2 * sqrt(S*(S+1)-ms[i]*ms[j])
        end
    end
    return Sx
end

function const_Sy(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sy = zeros(ComplexF64, dims, dims)
    for j in 1:dims, i in 1:dims
        if i-j == 1
            Sy[i,j] = -1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j])
        elseif j-i == 1
            Sy[i,j] =  1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j])
        end
    end
    return Sy
end

function const_Sz(S::Real)
    dims = Int(2*S + 1)
    ms = [S-i+1 for i in 1:dims]
    Sz = zeros(Float64, dims, dims)
    for i in 1:dims
        Sz[i,i] = ms[i]
    end
    return Sz
end

"""
    hamiltonian(model::HamiltonianModel)

Return the Hamiltonian for the given `model` as a two-site operator.
"""
function hamiltonian end

function hamiltonian_trunc(model::HamiltonianModel)
    h = hamiltonian(model)
    return hamiltonian_trunc(h)
end

function hamiltonian_trunc(h)
    d = size(h, 1)
    U, S, V = svd(reshape(h,d^2,d^2))
    truc = sum(S .> 1e-10)
    h1 = U[:,1:truc] * Diagonal(S[1:truc])
    h2 = V[:,1:truc]'
    return reshape(h1, d,d,truc), reshape(h2, truc,d,d)
end

function hamiltonian_trunc(model, direction)
    if direction=="right"
        h = hamiltonian_right(model)
    elseif direction=="down"
        h = hamiltonian_down(model)
    else
        error("Not implemented")
    end
    d = size(h, 1)
    U, S, V = svd(reshape(h,d^2,d^2))
    truc = sum(S .> 1e-10)
    h1 = U[:,1:truc] * Diagonal(S[1:truc])
    h2 = V[:,1:truc]'
    return reshape(h1, d,d,truc), reshape(h2, truc,d,d)
end
