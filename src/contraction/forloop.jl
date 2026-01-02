function forloop(f, args...; forloop_iter, N_in, N_out, size_out)
    if forloop_iter == 1
        return f(args...)
    else
        D_split = size(args[N_in[1]])[N_in[2]]
        result = similar(args[1], size_out)
        D_split_loop = cld(D_split, forloop_iter)
        D_split_ranges = [range(1 + (i-1)*D_split_loop, min(i*D_split_loop, D_split)) for i in 1:forloop_iter]

        for range in D_split_ranges
            cols_in = (j == N_in[2] ? range : (:) for j in 1:ndims(args[N_in[1]]))
            cols_out = (j == N_out ? range : (:) for j in 1: ndims(result))
            split_args = Tuple(j == N_in[1] ? args[j][cols_in...] : args[j] for j in 1:length(args))
            result[cols_out...] = f(split_args...)
        end

        return result
    end
end

function FLmap_forloop(FL, ALu, ALd, M; forloop_iter) 
    N_in = (3, ndims(ALd))
    N_out = ndims(ALd)
    χ = size(FL, 1)
    if M isa Tuple
        D1 = size(M[1], 3)
        D2 = size(M[2], 3)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 3)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 3)
        size_out = (χ,D,χ)
    end
    return forloop(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
end

function FRmap_forloop(FR, ARu, ARd, M; forloop_iter)
    N_in = (3, 1)
    N_out = ndims(ARd)
    χ = size(ARd, 1)
    if M isa Tuple
        D1 = size(M[1], 1)
        D2 = size(M[2], 1)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 1)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 1)
        size_out = (χ,D,χ)
    end
    return forloop(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out)
end

function ACmap_forloop(AC, FL, FR, M; forloop_iter)
    N_in = (3, ndims(FR))
    N_out = ndims(FR)
    χ = size(FR, 1)
    if M isa Tuple
        D1 = size(M[1], 2)
        D2 = size(M[2], 2)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 2)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 2)
        size_out = (χ,D,χ)
    end
    return forloop(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out)
end

function ACdmap_forloop(ACd, FL, FR, M; forloop_iter)
    N_in = (3, 1)
    N_out = ndims(FR)
    χ = size(FR, 1)
    if M isa Tuple
        D1 = size(M[1], 4)
        D2 = size(M[2], 4)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 4)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 4)
        size_out = (χ,D,χ)
    end
    return forloop(ACdmap, ACd, FL, FR, M; forloop_iter, N_in, N_out, size_out)
end

function forloop_sum(f, args...; forloop_iter, N_in1, N_in2, size_out)
    if forloop_iter == 1
        return f(args...)
    else
        D_split = size(args[N_in1[1]])[N_in1[2]]
        result = similar(args[1], size_out)
        result .= 0
        D_split_loop = cld(D_split, forloop_iter)
        D_split_ranges = [range(1 + (i-1)*D_split_loop, min(i*D_split_loop, D_split)) for i in 1:forloop_iter]

        for range in D_split_ranges
            cols_in1 = (j == N_in1[2] ? range : (:) for j in 1:ndims(args[N_in1[1]]))
            cols_in2 = (j == N_in2[2] ? range : (:) for j in 1:ndims(args[N_in2[1]]))
            split_args = (j == N_in1[1] ? args[j][cols_in1...] : (j == N_in2[1] ? args[j][cols_in2...] : args[j]) for j in 1:length(args))
            result += f(split_args...)
        end
        
        return result
    end
end

function Mumap_forloop(AC, ACd, FL, FR, Mu; forloop_iter)
    N_in1 = (2, 4)
    N_in2 = (4, 4)
    D1 = size(FL, 3)
    D2 = size(ACd, 3)
    D3 = size(FR, 3)
    D4 = size(AC, 3)
    
    d = size(Mu, 5)
    size_out = (D1,D2,D3,D4,d)
    return forloop_sum(Mumap, AC, ACd, FL, FR, Mu; forloop_iter, N_in1, N_in2, size_out)
end
