function forloop(f, args...; forloop_iter, N_in, N_out, size_out) #Problem: cannot AD by Zygote
    if forloop_iter == 1
        return f(args...)
    else
        D_split = size(args[N_in[1]])[N_in[2]]
        result = Zygote.Buffer(args[1], size_out)
        D_split_loop = cld(D_split, forloop_iter)
        D_split_ranges = [range(1 + (i-1)*D_split_loop, min(i*D_split_loop, D_split)) for i in 1:forloop_iter]
        for range in D_split_ranges
            cols_in = Zygote.@ignore (j == N_in[2] ? range : (:) for j in 1:ndims(args[N_in[1]]))
            cols_out = Zygote.@ignore (j == N_out ? range : (:) for j in 1: ndims(result))
            split_args = [j == N_in[1] ? args[j][cols_in...] : args[j] for j in 1:length(args)]
            result[cols_out...] = checkpoint(f, split_args...)
        end
        return copy(result)
    end
end

# function FLmap_forloop(FL, ALu, ALd, M; forloop_iter) 
#     N_in = (3, ndims(ALd))
#     N_out = ndims(ALd)
#     χ = size(FL, 1)
#     D = size(M, 3)
#     if ndims(M) == 4
#         size_out = (χ,D,χ)
#     else
#         size_out = (χ,D,D,χ)
#     end
#     return forloop(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
# end

function FLmap_forloop(FL, ALu, ALd, M; forloop_iter)
    if forloop_iter == 1
        return FLmap(FL, ALu, ALd, M)
    else
        χ1,χ2 = size(ALd)[[1,end]]
        if length(M) == 2
            D1 = size(M[1], 3)
            D2 = size(M[2], 3)
            FLm = Zygote.Buffer(FL, χ1,D1,D2,χ2)
        elseif ndims(M) == 5
            D1 = D2 = size(M, 3)
            FLm = Zygote.Buffer(FL, χ1,D1,D2,χ2)
        else
            D = size(M, 3)
            FLm = Zygote.Buffer(FL, χ1,D,χ2)
        end
        
        χ_loop = cld(χ2, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ2)) for i in 1:forloop_iter]
        cols = fill(:,ndims(FL)-1)
        for i in χ_ranges
            FLm[cols...,i] = checkpoint(FLmap, FL, ALu, ALd[cols...,i], M)
        end
        return copy(FLm)
    end
end

function FRmap_forloop(FR, ARu, ARd, M; forloop_iter) 
    if forloop_iter == 1
        return FRmap(FR, ARu, ARd, M)
    else
        χ1,χ2 = size(ARd)[[1,end]]
        if length(M) == 2
            D1 = size(M[1], 1)
            D2 = size(M[2], 1)
            FRm = Zygote.Buffer(FR, χ2,D1,D2,χ1)
        elseif ndims(M) == 5
            D1 = D2 = size(M, 1)
            FRm = Zygote.Buffer(FR, χ2,D1,D2,χ1)
        else
            D = size(M, 1)
            FRm = Zygote.Buffer(FR, χ2,D,χ1)
        end
        
        χ_loop = cld(χ1, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ1)) for i in 1:forloop_iter]
        cols = fill(:,ndims(FR)-1)
        for i in χ_ranges
            FRm[cols..., i] = checkpoint(FRmap, FR, ARu, ARd[i,cols...], M)
        end
        return copy(FRm)
    end
end

function ACmap_forloop(AC, FL, FR, M; forloop_iter) 
    if forloop_iter == 1
        return ACmap(AC, FL, FR, M)
    else
        χ1,χ2 = size(FR)[[1,end]]  
        if length(M) == 2
            D1 = size(M[1], 2)
            D2 = size(M[2], 2)
            ACm = Zygote.Buffer(AC, χ1,D1,D2,χ2)
        elseif ndims(M) == 5
            D1 = D2 = size(M, 2)
            ACm = Zygote.Buffer(AC, χ1,D1,D2,χ2)
        else
            D = size(M, 2)
            ACm = Zygote.Buffer(AC, χ1,D,χ2)
        end
        
        χ_loop = cld(χ2, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ2)) for i in 1:forloop_iter]
        cols = fill(:,ndims(AC)-1)
        for i in χ_ranges
            ACm[cols...,i] = checkpoint(ACmap, AC, FL, FR[cols...,i], M)
        end
        return copy(ACm)
    end
end