function FLmap(FL, ALu, ALd, M::Vector) 
    atype = _arraytype(FL[1])
    N_device = device_count(atype)
    χ = size(FL[1], 1)
    χ_device = cld(χ, N_device)
    χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:N_device]
    results = Vector{Any}(undef, N_device)
    FLm = copy(FL)

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                results[i] = FLmap(FL[i], ALu[i], view(ALd[i], :,:,χ_ranges[i]), M[i])
            end
        end
    end

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                for j in 1:N_device
                    FLm[i][:,:,χ_ranges[j]] .= atype(results[j])
                end
            end
        end
    end
    
    return FLm
end