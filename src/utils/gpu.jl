#helper functions to handle array types
_mattype(::Array) = Matrix
_mattype(::CuArray) = CuMatrix
_mattype(::ROCArray) = ROCMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::ROCArray) = ROCArray
_arraytype(S::StructArray) = _arraytype(S.data[1])

function ROCArray(x::NamedTuple)
    x.data .= map(ROCArray, x.data)
    return x
end

function CuArray(x::NamedTuple)
    x.data .= map(CuArray, x.data)
    return x
end

Array(x::NamedTuple) = x

function gc(::Type{Array})
    N_device = device_count(Array)
    for i in 1:N_device
        set_device_id!(Array, i)
        GC.gc()
    end
    @debug "GC triggered"
    return nothing
end

function gc(::Type{<:CuArray})
    N_device = device_count(CuArray)
    for i in 1:N_device
        set_device_id!(CuArray, i)
        GC.gc()
        CUDA.reclaim()
    end
    @debug "GC triggered"
    return nothing
end

function synchronize(x::AbstractArray)
    if x isa CuArray
        CUDA.synchronize()
    elseif x isa ROCArray
        AMDGPU.synchronize()
    end
end

function reclaim(x::AbstractArray)
    if x isa CuArray
        if CUDA.available_memory() / CUDA.total_memory() < 0.1
            GC.gc(true)
            CUDA.reclaim()
        end
    elseif x isa ROCArray
        GC.gc(true)
        AMDGPU.HIP.reclaim()
    end
end

for_gc(x) = x
function ChainRulesCore.rrule(::typeof(for_gc), x)
    function back(dx)
        reclaim(x[1])
        return NoTangent(), dx
    end
    return x, back
end
