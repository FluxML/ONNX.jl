
const ONNX2JULIA_TYPES = Dict(
    Integer(var"TensorProto.DataType".FLOAT) => Float32,
    Integer(var"TensorProto.DataType".UINT8) => UInt8,
    Integer(var"TensorProto.DataType".INT8) => Int8,
    Integer(var"TensorProto.DataType".UINT16) => UInt16,
    Integer(var"TensorProto.DataType".INT16) => Int16,
    Integer(var"TensorProto.DataType".INT32) => Int32,
    Integer(var"TensorProto.DataType".INT64) => Int64,
    Integer(var"TensorProto.DataType".STRING) => String,
    Integer(var"TensorProto.DataType".BOOL) => Bool,
    Integer(var"TensorProto.DataType".FLOAT16) => Float16,
    Integer(var"TensorProto.DataType".DOUBLE) => Float64,
    Integer(var"TensorProto.DataType".UINT32) => UInt32,
    Integer(var"TensorProto.DataType".UINT64) => UInt64,
    Integer(var"TensorProto.DataType".COMPLEX64) => Complex{Float32},
    Integer(var"TensorProto.DataType".COMPLEX128) => Complex{Float64},
    Integer(var"TensorProto.DataType".BFLOAT16) => Float16,
)

const ONNX2JULIA_DATA_FIELDS = Dict(
    Integer(var"TensorProto.DataType".INT64) => :int64_data,
    Integer(var"TensorProto.DataType".INT32) => :int32_data,
    # Integer(var"TensorProto.DataType".INT8) => no special field
    Integer(var"TensorProto.DataType".DOUBLE) => :double_data,
    Integer(var"TensorProto.DataType".FLOAT) => :float_data,
    # Integer(var"TensorProto.DataType".FLOAT16) => no special field
)

"""
    array(p::TensorProto, wrap=Array)

Return `p` as an `Array` of the correct type. Second argument can be used to change type of the returned array
"""
function array(p::TensorProto, wrap=Array)
    T = ONNX2JULIA_TYPES[p.data_type]
    fld = get(ONNX2JULIA_DATA_FIELDS, p.data_type, :raw_data)
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
