
function FLmap_forloop(FL, ALu, ALd, M; forloop_iter=1)
    if forloop_iter == 1
        return FLmap(FL, ALu, ALd, M)
    else
        χ = size(FL, 1)
        D = size(M, 3)
        if ndims(M) == 4
            FLm = Zygote.Buffer(FL, χ,D,χ)
        else
            FLm = Zygote.Buffer(FL, χ,D,D,χ)
        end
        χ_loop = cld(χ, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ)) for i in 1:forloop_iter]
        cols = fill(:,ndims(FL)-1)
        for i in χ_ranges
            FLm[cols...,i] = checkpoint(FLmap, FL, ALu, ALd[cols...,i], M)
        end
        return copy(FLm)
    end
end

function FLmap_forloop(FL, ALu, ALd, M1, M2; forloop_iter=1)
    if forloop_iter == 1
        return FLmap(FL, ALu, ALd, M1, M2)
    else
        χ = size(FL, 1)
        D = size(M1, 3)
        FLm = Zygote.Buffer(FL, χ,D,D,χ)
        χ_loop = cld(χ, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ)) for i in 1:forloop_iter]
        cols = fill(:,ndims(FL)-1)
        for i in χ_ranges
            FLm[cols...,i] = checkpoint(FLmap, FL, ALu, ALd[cols...,i], M1, M2)
        end
        return copy(FLm)
    end
end


function FRmap_forloop(FR, ARu, ARd, M; forloop_iter=1) 
    if forloop_iter == 1
        return FRmap(FR, ARu, ARd, M)
    else
        χ = size(FR, 1)
        D = size(M, 1)
        if ndims(M) == 4  
            FRm = Zygote.Buffer(FR, χ,D,χ)
        else
            FRm = Zygote.Buffer(FR, χ,D,D,χ)
        end
        χ_loop = cld(χ, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ)) for i in 1:forloop_iter]
        cols = fill(:,ndims(FR)-1)
        for i in χ_ranges
            FRm[i,cols...] = checkpoint(FRmap, FR, ARu[i,cols...], ARd, M)
        end
        return copy(FRm)
    end
end

function FRmap_forloop(FR, ARu, ARd, M1, M2; forloop_iter=1) 
    if forloop_iter == 1
        return FRmap(FR, ARu, ARd, M1, M2)
    else
        χ = size(FR, 1)
        D = size(M1, 1)
        FRm = Zygote.Buffer(FR, χ,D,D,χ)
        χ_loop = cld(χ, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ)) for i in 1:forloop_iter]
        cols = fill(:,ndims(FR)-1)
        for i in χ_ranges
            FRm[i,cols...] = checkpoint(FRmap, FR, ARu[i,cols...], ARd, M1, M2)
        end
        return copy(FRm)
    end
end



function ACmap_forloop(AC, FL, FR, M; forloop_iter=1) 
    if forloop_iter == 1
        return ACmap(AC, FL, FR, M)
    else
        χ = size(AC)[end]
        D = size(M, 2)
        if ndims(M) == 4
            ACm = Zygote.Buffer(AC, χ,D,χ)
        else
            ACm = Zygote.Buffer(AC, χ,D,D,χ)
        end
        χ_loop = cld(χ, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ)) for i in 1:forloop_iter]
        cols = fill(:,ndims(AC)-1)
        for i in χ_ranges
            ACm[cols...,i] = checkpoint(ACmap, AC,FL,FR[cols...,i],M)
        end
        return copy(ACm)
    end
end

function ACmap_forloop(AC, FL, FR, M1, M2; forloop_iter=1) 
    if forloop_iter == 1
        return ACmap(AC, FL, FR, M1, M2)
    else
        χ = size(AC)[end]
        D = size(M1, 2)
        ACm = Zygote.Buffer(AC, χ,D,D,χ)
        χ_loop = cld(χ, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ)) for i in 1:forloop_iter]
        cols = fill(:,ndims(AC)-1)
        for i in χ_ranges
            ACm[cols...,i] = checkpoint(ACmap, AC,FL,FR[cols...,i],M1,M2)
        end
        return copy(ACm)
    end
end
