#helper functions to handle array types
_arraytype(::Array{T}) where {T} = Array
_arraytype(::CuArray{T}) where {T} = CuArray
_arraytype(x::TensorMap) = _arraytype(x.data)

CuArray(t::TensorMap) = TensorMap(CuArray(t.data), t.space)
Array(t::TensorMap) = TensorMap(Array(t.data), t.space)