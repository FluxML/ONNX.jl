###############################################################################
#                              ValueInfoProto                                 #
###############################################################################

# julia shape type: vector or tuple of Integers or Missings
# used to dispatch to avoid signature collision with the default constructors
const JLShapeElem = Union{<:Integer, Missing}
const JLShape = Union{Tuple, Vector{<:JLShapeElem}}

const TypeProto_Tensor = var"TypeProto.Tensor"
const TensorShapeProto_Dimension = var"TensorShapeProto.Dimension"


TensorShapeProto_Dimension(::Missing) = var"TensorShapeProto.Dimension"()
TensorShapeProto_Dimension(n::Integer) = TensorShapeProto_Dimension(; value=OneOf(:dim_value, n))
TensorShapeProto_Dimension(s::String) = TensorShapeProto_Dimension(; value=OneOf(:dim_param, s))
TensorShapeProto_Dimension(s::Symbol) = TensorShapeProto_Dimension(string(s))


# note: do NOT rename to TensorShapeProto since it leads to complicated
# name collision with the default constructor
function to_tensor_shape_proto(shape::JLShape)
    return TensorShapeProto(dim=[TensorShapeProto_Dimension(s) for s in reverse(shape)])
end

dim_value(::Nothing) = missing
dim_value(v::OneOf) = (@assert v.name == :dim_value; v.value)
dim_value(d::TensorShapeProto_Dimension) = dim_value(d.value)

function Base.show(io::IO, tsp::TensorShapeProto)
    dim_str = join(map(dim_value, tsp.dim), ", ")
    print(io, "TensorShapeProto(dim(C)=($dim_str))")
end


elem_type_code(::Missing) = Integer(var"TensorProto.DataType".UNDEFINED)
elem_type_code(::Type{Float32}) = Integer(var"TensorProto.DataType".FLOAT)
elem_type_code(::Type{UInt8}) = Integer(var"TensorProto.DataType".UINT8)
elem_type_code(::Type{Int8}) = Integer(var"TensorProto.DataType".INT8)
elem_type_code(::Type{UInt16}) = Integer(var"TensorProto.DataType".UINT16)
elem_type_code(::Type{Int16}) = Integer(var"TensorProto.DataType".INT16)
elem_type_code(::Type{Int32}) = Integer(var"TensorProto.DataType".INT32)
elem_type_code(::Type{Int64}) = Integer(var"TensorProto.DataType".INT64)
elem_type_code(::Type{String}) = Integer(var"TensorProto.DataType".STRING)
elem_type_code(::Type{Bool}) = Integer(var"TensorProto.DataType".BOOL)
elem_type_code(::Type{Float16}) = Integer(var"TensorProto.DataType".FLOAT16)
elem_type_code(::Type{Float64}) = Integer(var"TensorProto.DataType".DOUBLE)
elem_type_code(::Type{UInt32}) = Integer(var"TensorProto.DataType".UINT32)
elem_type_code(::Type{UInt64}) = Integer(var"TensorProto.DataType".UINT64)
#elem_type_code(::Type{Complex{Float32}}) = Integer(var"TensorProto.DataType".COMPLEX64)
#elem_type_code(::Type{Complex{Float64}}) = Integer(var"TensorProto.DataType".COMPLEX128)


function ValueInfoProto(name::String, shape::JLShape, jltyp=Float32)
    tp_t = TypeProto_Tensor(elem_type_code(jltyp), to_tensor_shape_proto(shape))
    tp = TypeProto(; value=OneOf(:tensor_type, tp_t))
    return ValueInfoProto(name=name, var"#type"=tp)
end

function Base.show(io::IO, vip::ValueInfoProto)
    tsp = vip.var"#type".value.value.shape
    print(io, "ValueInfoProto($(vip.name), $tsp)")
end

# TODO: User supplied elemtype??
# function ValueInfoProto(name::String, inshape, elemtype=Float32)
#     tp_t = var"TypeProto.Tensor"(elemtype, inshape)
#     tp = TypeProto(; value=OneOf(:tensor_type, tp_t))
#     ValueInfoProto(name=name, var"#type"=tp)
# end


# # i.e. (elemtype, shape)
# var"TypeProto.Tensor"(elemtype, inshape) = var"TypeProto.Tensor"(
#     elem_type=tp_tensor_elemtype(elemtype),
#     shape=tensor_shape_proto(inshape)
# )
# var"TypeProto.Tensor"(elemtype, ::Missing) = var"TypeProto.Tensor"(
#     elem_type=tp_tensor_elemtype(elemtype)
# )

# tensor_shape_proto(shape) = TensorShapeProto(dim=[tsp_d(s) for s in reverse(shape)])
# tsp_d(::Missing) = var"TensorShapeProto.Dimension"()
# # tsp_d(n::Integer) = var"TensorShapeProto.Dimension"(dim_value=n)
# # tsp_d(s::String) = var"TensorShapeProto.Dimension"(dim_param=s)
# tsp_d(n::Integer) = var"TensorShapeProto.Dimension"(; value=OneOf(:dim_value, n))
# tsp_d(s::String) = var"TensorShapeProto.Dimension"(; value=OneOf(:dim_param, s))
# tsp_d(s::Symbol) = tsp_d(string(s))

# tp_tensor_elemtype(i::Integer) = i
# # # TODO: probably we need to convert these values to Integer as well
# # tp_tensor_elemtype(::Missing) = var"TensorProto.DataType".UNDEFINED
# # tp_tensor_elemtype(::Type{Int32}) = var"TensorProto.DataType".INT32
# # tp_tensor_elemtype(::Type{Int64}) = var"TensorProto.DataType".INT64
# # tp_tensor_elemtype(::Type{Float32}) = var"TensorProto.DataType".FLOAT
# # tp_tensor_elemtype(::Type{Float64}) = var"TensorProto.DataType".DOUBLE
# # TODO: probably we need to convert these values to Integer as well
# tp_tensor_elemtype(::Missing) = Integer(var"TensorProto.DataType".UNDEFINED)
# tp_tensor_elemtype(::Type{Int32}) = Integer(var"TensorProto.DataType".INT32)
# tp_tensor_elemtype(::Type{Int64}) = Integer(var"TensorProto.DataType".INT64)
# tp_tensor_elemtype(::Type{Float32}) = Integer(var"TensorProto.DataType".FLOAT)
# tp_tensor_elemtype(::Type{Float64}) = Integer(var"TensorProto.DataType".DOUBLE)

TensorProto(x::Number, name ="") = TensorProto([x], name)

TensorProto(t::AbstractArray{Float64,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type = elem_type_code(Float64),
    double_data = reshape(t, :),
    name=name)

TensorProto(t::AbstractArray{Float32,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=elem_type_code(Float32),
    float_data = reshape(t, :),
    name=name)

TensorProto(t::AbstractArray{Float16,N}, name ="") where N = TensorProto(t, elem_type_code(Float16), name)

TensorProto(t::AbstractArray{Int64,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=elem_type_code(Int64),
    int64_data = reshape(t, :),
    name=name)

TensorProto(t::AbstractArray{Int32,N}, name ="") where N = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=elem_type_code(Int32),
    int32_data = reshape(t, :),
    name=name)

TensorProto(t::AbstractArray{Int8,N}, name ="") where N = TensorProto(t, elem_type_code(Int8), name)

TensorProto(t::AbstractArray, data_type::Int32, name) = TensorProto(
    dims=collect(reverse(size(t))),
    data_type=data_type,
    raw_data = reinterpret(UInt8, reshape(t, :)),
    name=name)


function Base.show(io::IO, a::AttributeProto)
    print(io, "AttributeProto($(attribute(a)))")
end

AttributeProto(p::Pair) = AttributeProto(first(p), last(p))
AttributeProto(name::Symbol, v) = AttributeProto(string(name), v)

AttributeProto(name::String, i::Int64) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".INT,
    i = i
)

AttributeProto(name::String, f::Float32) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".FLOAT,
    f = f
)

AttributeProto(name::String, floats::AbstractVector{Float32}) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".FLOATS,
    floats = floats
)

AttributeProto(name::String, f::Float64) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".FLOAT,
    f = Float32(f)
)

AttributeProto(name::String, i::NTuple{N, Int64}) where N = AttributeProto(name, collect(i))

AttributeProto(name::String, i::AbstractVector{Int64}) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".INTS,
    ints = i
)

AttributeProto(name::String, str::AbstractString) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".STRING,
    s = Vector{UInt8}(str)
)

AttributeProto(name::String, strings::AbstractVector{<:AbstractString}) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".STRINGS,
    strings = Vector{UInt8}.(strings)
)

AttributeProto(name::String, tensor::TensorProto) = AttributeProto(
    name=name,
    var"#type" = var"AttributeProto.AttributeType".TENSOR,
    t = tensor
)
