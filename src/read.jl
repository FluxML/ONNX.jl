
const ONNX2JULIA_TYPES = Dict(
    TensorProto_DataType.INT64 => Int64,
    TensorProto_DataType.INT32 => Int32,
    TensorProto_DataType.INT8 => Int8,
    TensorProto_DataType.DOUBLE => Float64,
    TensorProto_DataType.FLOAT => Float32,
    TensorProto_DataType.FLOAT16 => Float16,
)

const ONNX2JULIA_DATA_FIELDS = Dict(
    TensorProto_DataType.INT64 => :int64_data,
    TensorProto_DataType.INT32 => :int32_data,
    # TensorProto_DataType.INT8 => no special field
    TensorProto_DataType.DOUBLE => :double_data,
    TensorProto_DataType.FLOAT => :float_data,
    # TensorProto_DataType.FLOAT16 => no special field
)

"""
    array(p::TensorProto)

Return `p` as an `Array` of the correct type. Second argument can be used to change type of the returned array
"""
function array(p::TensorProto, wrap=Array)
    T = ONNX2JULIA_TYPES[p.data_type]
    fld = get(ONNX2JULIA_DATA_FIELDS, p.data_type, :raw_data)
    bytes = getproperty(p, fld)
    data = !isempty(bytes) ? reinterpret(T, bytes) : reinterpret(T, p.raw_data)
    return reshape(wrap(data), reverse(p.dims)...)
end

Base.size(vip::ValueInfoProto) = size(vip._type)
Base.size(tp::TypeProto) = size(tp.tensor_type)
Base.size(tp::TensorProto) = tp.dims
Base.size(tp_t::TypeProto_Tensor) = hasproperty(tp_t, :shape) ? size(tp_t.shape) : missing
Base.size(tsp::TensorShapeProto) = size.(Tuple(reverse(tsp.dim)))
Base.size(tsp_d::TensorShapeProto_Dimension) = hasproperty(tsp_d, :dim_value) ? tsp_d.dim_value : missing

"""
    attribute(p::AttributeProto)

Return attribute in `p` as a name => value pair.
"""
function attribute(p::AttributeProto)
    # Copy paste from ONNX.jl
    if (p._type != 0)
        field = [:f, :i, :s, :t, :g, :floats, :ints, :strings, :tensors, :graphs][p._type]
        if field === :s
            return Symbol(p.name) => String(getproperty(p, field))
        elseif  field === :strings
            return Symbol(p.name) => String.(getproperty(p, field))
        end
        return Symbol(p.name) => getproperty(p, field)
    end
end

Base.Dict(pa::AbstractVector{AttributeProto}) = Dict(attribute(p) for p in pa)
