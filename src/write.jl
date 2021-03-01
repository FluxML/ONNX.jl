
# TODO: User supplied elemtype??
ValueInfoProto(name::String, inshape, elemtype=Float32) =
ValueInfoProto(
    name=name,
    _type=TypeProto(
        tensor_type=TypeProto_Tensor(inshape, elemtype)
    )
)

TypeProto_Tensor(inshape, elemtype) = TypeProto_Tensor(
    elem_type=tp_tensor_elemtype(elemtype),
    shape=TensorShapeProto(inshape)
)
TypeProto_Tensor(::Missing, elemtype) = TypeProto_Tensor(
    elem_type=tp_tensor_elemtype(elemtype)
)

TensorShapeProto(shape) = TensorShapeProto(dim=[tsp_d(s) for s in reverse(shape)])
tsp_d(::Missing) = TensorShapeProto_Dimension()
tsp_d(n::Integer) = TensorShapeProto_Dimension(dim_value=n)
tsp_d(s::String) = TensorShapeProto_Dimension(dim_param=s)
tsp_d(s::Symbol) = tsp_d(string(s))

tp_tensor_elemtype(i::Integer) = i
tp_tensor_elemtype(::Missing) = TensorProto_DataType.UNDEFINED
tp_tensor_elemtype(::Type{Float32}) = TensorProto_DataType.FLOAT

TensorProto(x::Number, name ="") = TensorProto([x], name)

TensorProto(t::AbstractArray{Float64,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=TensorProto_DataType.DOUBLE,
    double_data = reshape(t,:),
    name=name)

TensorProto(t::AbstractArray{Float32,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=TensorProto_DataType.FLOAT,
    float_data = reshape(t,:),
    name=name)

TensorProto(t::AbstractArray{Float16,N}, name ="") where N = TensorProto(t, TensorProto_DataType.FLOAT16, name)

TensorProto(t::AbstractArray{Int64,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=TensorProto_DataType.INT64,
    int64_data = reshape(t,:),
    name=name)

TensorProto(t::AbstractArray{Int32,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=TensorProto_DataType.INT32,
    int32_data = reshape(t,:),
    name=name)

TensorProto(t::AbstractArray{Int8,N}, name ="") where N = TensorProto(t, TensorProto_DataType.INT8, name)

TensorProto(t::AbstractArray, data_type::Int32, name) = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=data_type,
    raw_data = reinterpret(UInt8, reshape(t,:)),
    name=name)


AttributeProto(p::Pair) = AttributeProto(first(p), last(p))
AttributeProto(name::Symbol, v) = AttributeProto(string(name), v)

AttributeProto(name::String, i::Int64) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.INT,
    i = i
)

AttributeProto(name::String, f::Float32) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.FLOAT,
    f = f
)

AttributeProto(name::String, floats::AbstractVector{Float32}) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.FLOATS,
    floats = floats
)

AttributeProto(name::String, f::Float64) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.FLOAT,
    f = Float32(f)
)

AttributeProto(name::String, i::NTuple{N, Int64}) where N = AttributeProto(name, collect(i))

AttributeProto(name::String, i::AbstractVector{Int64}) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.INTS,
    ints = i
)

AttributeProto(name::String, str::AbstractString) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.STRING,
    s = Vector{UInt8}(str)
)

AttributeProto(name::String, strings::AbstractVector{<:AbstractString}) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.STRINGS,
    strings = Vector{UInt8}.(strings)
)

AttributeProto(name::String, tensor::TensorProto) = AttributeProto(
    name=name,
    _type = AttributeProto_AttributeType.TENSOR,
    t = tensor
)
