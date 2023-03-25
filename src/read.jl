
onnx2julia_types(x::T) where T =
    @error "Unknown type $(x), check at onnx.proto3 (message TensorProto)"
onnx2julia_types(data_type::Integer) = onnx2julia_types(Val(Int32(data_type)))
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".FLOAT)}) = Float32
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".UINT8)}) = UInt8
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".INT8)}) = Int8
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".UINT16)}) = UInt16
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".INT16)}) = Int16
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".INT32)}) = Int32
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".INT64)}) = Int64
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".STRING)}) = String
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".BOOL)}) = Bool
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".FLOAT16)}) = Float16
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".DOUBLE)}) = Float64
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".UINT32)}) = UInt32
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".UINT64)}) = UInt64
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".COMPLEX64)}) = Complex{Float32}
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".COMPLEX128)}) = Complex{Float64}
onnx2julia_types(::Val{Integer(var"TensorProto.DataType".BFLOAT16)}) = @error "BFloat16 isn't supported yet"


onnx2julia_data_fields(x::T) where T =
    @error "Unknown type $(x), check at onnx.proto3 (message TensorProto)"
onnx2julia_data_fields(data_type::Integer) = onnx2julia_data_fields(Val(Int32(data_type)))
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".FLOAT)}) = :float_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".UINT8)}) = :raw_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".INT8)}) = :raw_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".UINT16)}) = :raw_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".INT16)}) = :raw_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".INT32)}) = :int32_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".INT64)}) = :int64_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".STRING)}) = :string_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".BOOL)}) = :raw_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".FLOAT16)}) = :raw_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".DOUBLE)}) = :double_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".UINT32)}) = :uint64_data
onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".UINT64)}) = :uint64_data
#onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".COMPLEX64)}) = 
#onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".COMPLEX128)}) = 
#onnx2julia_data_fields(::Val{Integer(var"TensorProto.DataType".BFLOAT16)}) = @error "BFloat16 is support yet"

"""
    array(p::TensorProto, wrap=Array)

Return `p` as an `Array` of the correct type. Second argument can be used to change type of the returned array
"""
function array(p::TensorProto, wrap=Array)
    T = onnx2julia_types(p.data_type)
    fld = onnx2julia_data_fields(p.data_type)
    bytes = getproperty(p, fld)
    data = !isempty(bytes) ? reinterpret(T, bytes) : reinterpret(T, p.raw_data)
    # note that we don't permute dimensions here, only reshape the data
    # see "Row-/Columns-major" in test/readwrite.jl for an example
    # why we don't need permutedims here
    return reshape(wrap(data), reverse(p.dims)...)
end

Base.size(vip::ValueInfoProto) = size(vip.var"#type")
Base.size(tp::TypeProto) = (@assert tp.value.name == :tensor_type; size(tp.value.value))
Base.size(tp::TensorProto) = tp.dims
Base.size(tp_t::var"TypeProto.Tensor") = hasproperty(tp_t, :shape) ? size(tp_t.shape) : missing
Base.size(tsp::TensorShapeProto) = size.(Tuple(reverse(tsp.dim)))
# Base.size(tsp_d::var"TensorShapeProto.Dimension") = hasproperty(tsp_d, :dim_value) ? tsp_d.dim_value : missing
function Base.size(tsp_d::var"TensorShapeProto.Dimension")
    isnothing(tsp_d.value) && return missing
    @assert tsp_d.value.name == :dim_value
    return tsp_d.value.value
end

"""
    attribute(p::AttributeProto)

Return attribute in `p` as a name => value pair.
"""
function attribute(p::AttributeProto)
    # Copy paste from ONNX.jl
    typ = Integer(p.var"#type")
    if (typ != 0)
        field = [:f, :i, :s, :t, :g, :floats, :ints, :strings, :tensors, :graphs][typ]
        if field === :s
            return Symbol(p.name) => String(getproperty(p, field))
        elseif  field === :strings
            return Symbol(p.name) => String.(getproperty(p, field))
        end
        return Symbol(p.name) => getproperty(p, field)
    end
end

Base.Dict(pa::AbstractVector{<:AttributeProto}) = Dict(attribute(p) for p in pa)
