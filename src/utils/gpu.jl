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

function gc(::Type{<:Array}; threshold::Real = 0.1)
    # Same gate as the GPU branches: fire only when free RAM ratio drops below
    # `threshold`. `threshold = Inf` forces unconditional collection.
    # Note: on Linux, `Sys.free_memory()` reports MemFree (excluding reclaimable
    # caches), so the ratio reads low even with plenty of usable memory — raise
    # the threshold or pass `Inf` if you want the same effective behaviour as
    # Windows, where Sys.free_memory() tracks MemAvailable.
    if threshold !== Inf && Sys.free_memory() / Sys.total_memory() >= threshold
        return nothing
    end
    N_device = device_count(Array)
    for i in 1:N_device
        set_device_id!(Array, i)
        GC.gc(true)               # full collection — match GPU branch
    end
    @debug "GC triggered (CPU, full)"
    return nothing
end

# Threshold-gated GPU GC + pool reclaim.
#   threshold ∈ (0, 1]: fire only when available_memory/total_memory < threshold
#   threshold = Inf:    unconditional (every call)
# Default 0.1 = trigger once pool is >90% reserved; pass higher (or Inf) in
# tight hot loops where you want forced cleanup every iter.
function gc(::Type{<:CuArray}; threshold::Real = 0.1)
    if threshold !== Inf && CUDA.available_memory() / CUDA.total_memory() >= threshold
        return nothing   # pool has headroom — skip
    end
    N_device = device_count(CuArray)
    for i in 1:N_device
        set_device_id!(CuArray, i)
        GC.gc(true)               # full collection, not incremental
        CUDA.reclaim()
    end
    @debug "GC triggered (CuArray, full)"
    return nothing
end

function gc(::Type{<:ROCArray}; threshold::Real = 0.1)
    # AMDGPU has no direct free-mem query analogous to CUDA.available_memory;
    # honour the threshold only when caller passes Inf (force) — otherwise
    # always fire (legacy behaviour).
    GC.gc(true)
    AMDGPU.HIP.reclaim()
    @debug "GC triggered (ROCArray, full)"
    return nothing
end

# Convenience: pass an array, dispatch on its type.
gc(x::AbstractArray; threshold::Real = 0.1) = gc(typeof(x); threshold)

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
