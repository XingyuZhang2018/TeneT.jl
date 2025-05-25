function FLmap_parallel(FL, ALu, ALd, M) 
    atype = _arraytype(FL)
    N_device = device_count(atype)
    χ = size(FL, 1)
    χ_device = cld(χ, N_device)
    χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:N_device]
    cols = fill(:,ndims(FL)-1)
    results = Vector{Any}(undef, N_device)

    set_device_id!(atype, 1)
    FLm = similar(FL)
    FLs = to_N_device(FL)
    ALus = to_N_device(ALu)
    ALds = to_N_device(ALd)
    Ms = to_N_device(M)

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                results[i] = FLmap(FLs[i], ALus[i], ALds[i][cols...,χ_ranges[i]], Ms[i])
            end
        end
    end

    set_device_id!(atype, 1)
    @sync begin
        for i in 1:N_device
            @async begin
                FLm[cols...,χ_ranges[i]] .= atype(results[i])
            end
        end
    end
    
    return FLm
end

function FRmap_parallel(FR, ARu, ARd, M) 
    atype = _arraytype(FR)
    N_device = device_count(atype)
    χ = size(FR, 1)
    χ_device = cld(χ, N_device)
    χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:N_device]
    cols = fill(:,ndims(FR)-1)
    results = Vector{Any}(undef, N_device)

    set_device_id!(atype, 1)
    FRm = similar(FR)
    FRs = to_N_device(FR)
    ARus = to_N_device(ARu)
    ARds = to_N_device(ARd)
    Ms = to_N_device(M)

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                results[i] = FRmap(FRs[i], ARus[i][χ_ranges[i],cols...], ARds[i], Ms[i])
            end
        end
    end

    set_device_id!(atype, 1)
    @sync begin
        for i in 1:N_device
            @async begin
                FRm[χ_ranges[i],cols...] .= atype(results[i])
            end
        end
    end
    
    return FRm
end

function ACmap_parallel(AC, FL, FR, M) 
    atype = _arraytype(AC)
    N_device = device_count(atype)
    χ = size(AC, 1)
    χ_device = cld(χ, N_device)
    χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:N_device]
    cols = fill(:,ndims(AC)-1)
    results = Vector{Any}(undef, N_device)

    set_device_id!(atype, 1)
    if ndims(M) == 4
        D = size(M,2)
        ACm = similar(AC, χ, D, χ)
    else
        D = size(M,3)
        ACm = similar(AC, χ, D, D, χ)
    end
    
    ACs = to_N_device(AC)
    FLs = to_N_device(FL)
    FRs = to_N_device(FR)
    Ms = to_N_device(M)

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                results[i] = ACmap(ACs[i], FLs[i], FRs[i][cols...,χ_ranges[i]], Ms[i])
            end
        end
    end

    set_device_id!(atype, 1)
    @sync begin
        for i in 1:N_device
            @async begin
                ACm[cols...,χ_ranges[i]] .= atype(results[i])
            end
        end
    end
    
    return ACm
end