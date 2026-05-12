#helper functions to handle array types
_mattype(::Array) = Matrix
_mattype(::CuArray) = CuMatrix
_mattype(::ROCArray) = ROCMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::ROCArray) = ROCArray
_arraytype(S::StructArray) = _arraytype(S.data[1])

set_device_id!(::Type{ROCArray}, i::Int) = AMDGPU.device_id!(i)
set_device_id!(::Type{CuArray}, i::Int) = CUDA.device!(i-1)
set_device_id!(::Type{Array}, i::Int) = task_local_storage(:cpu_device_id, i)

get_device(::Type{ROCArray}) = AMDGPU.device()
get_device(::Type{CuArray}) = CUDA.device()
get_device(::Type{Array}) = "CPU task $(get_device_id(Array))"

device_count(::Type{ROCArray}) = length(AMDGPU.devices())
device_count(::Type{CuArray}) = length(CUDA.devices())
device_count(::Type{Array}) = Threads.nthreads()

get_device_id(::Array) = get(task_local_storage(), :cpu_device_id, 1)
get_device_id(x::ROCArray) = Int(AMDGPU.device(x).device_id)
get_device_id(x::CuArray) = Int(CUDA.device(x).handle + 1)
get_device_id(S::StructArray) = get_device_id(S[1])

get_device_id(::Type{Array}) = get(task_local_storage(), :cpu_device_id, 1)
get_device_id(::Type{ROCArray}) = AMDGPU.device_id()
get_device_id(::Type{CuArray}) = Int(CUDA.device().handle) + 1

function atype_device!(atype, x, i::Int)
    set_device_id!(atype, i)
    return atype(x)
end

function ROCArray(x::NamedTuple)
    x.data .= map(ROCArray, x.data)
    return x
end

function CuArray(x::NamedTuple)
    x.data .= map(CuArray, x.data)
    return x
end

Array(x::NamedTuple) = x
CuArray(::Nothing) = nothing

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
        GC.gc(true)              # full collection, not incremental
        CUDA.reclaim()
    end
    @debug "GC triggered (CuArray, full)"
    return nothing
end

function gc(::Type{<:ROCArray})
    GC.gc(true)
    AMDGPU.HIP.reclaim()
    @debug "GC triggered (ROCArray, full)"
    return nothing
end

# Convenience: pass an array, dispatch on its type. Always unconditional —
# callers that want conditional cleanup should gate themselves.
gc(x::AbstractArray) = gc(typeof(x))

function synchronize(x::AbstractArray)
    if x isa CuArray
        CUDA.synchronize()
    elseif x isa ROCArray
        AMDGPU.synchronize()
    end
end

for_gc(x) = x
function ChainRulesCore.rrule(::typeof(for_gc), x)
    function back(dx)
        gc(x[1])
        return NoTangent(), dx
    end
    return x, back
end
