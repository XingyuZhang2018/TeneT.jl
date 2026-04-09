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

function const_Sp(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sp = zeros(Float64, dims, dims)
    for j in 1:dims, i in 1:dims
        if i-j == 1
            Sp[i,j] = sqrt(S*(S+1)-ms[i]*ms[j])
        end
    end
    return Sp
end

function const_Sm(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sm = zeros(Float64, dims, dims)
    for j in 1:dims, i in 1:dims
        if j-i == 1
            Sm[i,j] = sqrt(S*(S+1)-ms[i]*ms[j])
        end
    end
    return Sm
end

