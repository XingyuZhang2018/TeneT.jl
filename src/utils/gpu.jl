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
set_device_id!(::Type{Array}, i::Int) = nothing

get_device(::Type{ROCArray}) = AMDGPU.device()
get_device(::Type{CuArray}) = CUDA.device()
get_device(::Type{Array}) = "CPU thread $(threadid())"

device_count(::Type{ROCArray}) = length(AMDGPU.devices())
device_count(::Type{CuArray}) = length(CUDA.devices())
device_count(::Type{Array}) = Threads.nthreads()

get_device_id(::Array) = 1
get_device_id(x::ROCArray) = Int(AMDGPU.device(x).device_id)
get_device_id(x::CuArray) = Int(CUDA.device(x).handle + 1)
get_device_id(S::StructArray) = get_device_id(S[1])

get_device_id(::Type{Array}) = Threads.threadid()
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

function gc(atype)
    N_device = device_count(atype)
    # @sync begin
        println("GC!")
        for i in 1:N_device
            # @async begin
                set_device_id!(atype, i)
                GC.gc()
                CUDA.reclaim()
            # end
        end
    # end
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
        GC.gc(true)
        CUDA.reclaim()
    elseif x isa ROCArray
        GC.gc(true)
        AMDGPU.HIP.reclaim()
    end
end

for_gc(x) = x
function ChainRulesCore.rrule(::typeof(for_gc), x)
    function back(dx)
        reclaim(x[1])
        return NoTangent, dx
    end
    return x, back
end
