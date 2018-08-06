# Generate protoBuf code, donot change directly.

module Proto

# syntax: proto2
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

struct __enum_Version <: ProtoEnum
    _START_VERSION::Int32
    IR_VERSION_2017_10_10::Int32
    IR_VERSION_2017_10_30::Int32
    IR_VERSION::Int32
    __enum_Version() = new(0,1,2,3)
end #struct __enum_Version
const Version = __enum_Version()

mutable struct StringStringEntryProto <: ProtoType
    key::AbstractString
    value::AbstractString
    StringStringEntryProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct StringStringEntryProto
hash(v::StringStringEntryProto) = ProtoBuf.protohash(v)
isequal(v1::StringStringEntryProto, v2::StringStringEntryProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::StringStringEntryProto, v2::StringStringEntryProto) = ProtoBuf.protoeq(v1, v2)

struct __enum_TensorProto_DataType <: ProtoEnum
    UNDEFINED::Int32
    FLOAT::Int32
    UINT8::Int32
    INT8::Int32
    UINT16::Int32
    INT16::Int32
    INT32::Int32
    INT64::Int32
    STRING::Int32
    BOOL::Int32
    FLOAT16::Int32
    DOUBLE::Int32
    UINT32::Int32
    UINT64::Int32
    COMPLEX64::Int32
    COMPLEX128::Int32
    __enum_TensorProto_DataType() = new(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
end #struct __enum_TensorProto_DataType
const TensorProto_DataType = __enum_TensorProto_DataType()

mutable struct TensorProto_Segment <: ProtoType
    _begin::Int64
    _end::Int64
    TensorProto_Segment(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorProto_Segment
hash(v::TensorProto_Segment) = ProtoBuf.protohash(v)
isequal(v1::TensorProto_Segment, v2::TensorProto_Segment) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorProto_Segment, v2::TensorProto_Segment) = ProtoBuf.protoeq(v1, v2)

mutable struct TensorProto <: ProtoType
    dims::Vector{Int64}
    data_type::Int32
    segment::TensorProto_Segment
    float_data::Vector{Float32}
    int32_data::Vector{Int32}
    string_data::Vector{Array{UInt8,1}}
    int64_data::Vector{Int64}
    name::AbstractString
    doc_string::AbstractString
    raw_data::Array{UInt8,1}
    double_data::Vector{Float64}
    uint64_data::Vector{UInt64}
    TensorProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorProto
const __fnum_TensorProto = Int[1,2,3,4,5,6,7,8,12,9,10,11]
const __pack_TensorProto = Symbol[:float_data,:int32_data,:int64_data,:double_data,:uint64_data]
meta(t::Type{TensorProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorProto, ProtoBuf.DEF_VAL, true, __pack_TensorProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::TensorProto) = ProtoBuf.protohash(v)
isequal(v1::TensorProto, v2::TensorProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorProto, v2::TensorProto) = ProtoBuf.protoeq(v1, v2)

mutable struct TensorShapeProto_Dimension <: ProtoType
    dim_value::Int64
    dim_param::AbstractString
    TensorShapeProto_Dimension(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorShapeProto_Dimension
const __oneofs_TensorShapeProto_Dimension = Int[1,1]
const __oneof_names_TensorShapeProto_Dimension = [Symbol("value")]
meta(t::Type{TensorShapeProto_Dimension}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_TensorShapeProto_Dimension, __oneof_names_TensorShapeProto_Dimension, ProtoBuf.DEF_FIELD_TYPES)
hash(v::TensorShapeProto_Dimension) = ProtoBuf.protohash(v)
isequal(v1::TensorShapeProto_Dimension, v2::TensorShapeProto_Dimension) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorShapeProto_Dimension, v2::TensorShapeProto_Dimension) = ProtoBuf.protoeq(v1, v2)

mutable struct TensorShapeProto <: ProtoType
    dim::Vector{TensorShapeProto_Dimension}
    TensorShapeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorShapeProto
hash(v::TensorShapeProto) = ProtoBuf.protohash(v)
isequal(v1::TensorShapeProto, v2::TensorShapeProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorShapeProto, v2::TensorShapeProto) = ProtoBuf.protoeq(v1, v2)

mutable struct TypeProto_Tensor <: ProtoType
    elem_type::Int32
    shape::TensorShapeProto
    TypeProto_Tensor(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TypeProto_Tensor
hash(v::TypeProto_Tensor) = ProtoBuf.protohash(v)
isequal(v1::TypeProto_Tensor, v2::TypeProto_Tensor) = ProtoBuf.protoisequal(v1, v2)
==(v1::TypeProto_Tensor, v2::TypeProto_Tensor) = ProtoBuf.protoeq(v1, v2)

mutable struct TypeProto <: ProtoType
    tensor_type::TypeProto_Tensor
    TypeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TypeProto
const __oneofs_TypeProto = Int[1]
const __oneof_names_TypeProto = [Symbol("value")]
meta(t::Type{TypeProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_TypeProto, __oneof_names_TypeProto, ProtoBuf.DEF_FIELD_TYPES)
hash(v::TypeProto) = ProtoBuf.protohash(v)
isequal(v1::TypeProto, v2::TypeProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::TypeProto, v2::TypeProto) = ProtoBuf.protoeq(v1, v2)

mutable struct ValueInfoProto <: ProtoType
    name::AbstractString
    _type::TypeProto
    doc_string::AbstractString
    ValueInfoProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ValueInfoProto
hash(v::ValueInfoProto) = ProtoBuf.protohash(v)
isequal(v1::ValueInfoProto, v2::ValueInfoProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::ValueInfoProto, v2::ValueInfoProto) = ProtoBuf.protoeq(v1, v2)

mutable struct OperatorSetIdProto <: ProtoType
    domain::AbstractString
    version::Int64
    OperatorSetIdProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OperatorSetIdProto
hash(v::OperatorSetIdProto) = ProtoBuf.protohash(v)
isequal(v1::OperatorSetIdProto, v2::OperatorSetIdProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::OperatorSetIdProto, v2::OperatorSetIdProto) = ProtoBuf.protoeq(v1, v2)

struct __enum_AttributeProto_AttributeType <: ProtoEnum
    UNDEFINED::Int32
    FLOAT::Int32
    INT::Int32
    STRING::Int32
    TENSOR::Int32
    GRAPH::Int32
    FLOATS::Int32
    INTS::Int32
    STRINGS::Int32
    TENSORS::Int32
    GRAPHS::Int32
    __enum_AttributeProto_AttributeType() = new(0,1,2,3,4,5,6,7,8,9,10)
end #struct __enum_AttributeProto_AttributeType
const AttributeProto_AttributeType = __enum_AttributeProto_AttributeType()

mutable struct AttributeProto <: ProtoType
    name::AbstractString
    doc_string::AbstractString
    _type::Int32
    f::Float32
    i::Int64
    s::Array{UInt8,1}
    t::TensorProto
    g::Any
    floats::Vector{Float32}
    ints::Vector{Int64}
    strings::Vector{Array{UInt8,1}}
    tensors::Vector{TensorProto}
    graphs::Any
    AttributeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AttributeProto
const __fnum_AttributeProto = Int[1,13,20,2,3,4,5,6,7,8,9,10,11]
const __ftype_AttributeProto = Dict(:g => "GraphProto", :graphs => "Vector{GraphProto}")
meta(t::Type{AttributeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_AttributeProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_AttributeProto)
hash(v::AttributeProto) = ProtoBuf.protohash(v)
isequal(v1::AttributeProto, v2::AttributeProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::AttributeProto, v2::AttributeProto) = ProtoBuf.protoeq(v1, v2)

mutable struct NodeProto <: ProtoType
    input::Vector{AbstractString}
    output::Vector{AbstractString}
    name::AbstractString
    op_type::AbstractString
    domain::AbstractString
    attribute::Vector{AttributeProto}
    doc_string::AbstractString
    NodeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NodeProto
const __fnum_NodeProto = Int[1,2,3,4,7,5,6]
meta(t::Type{NodeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_NodeProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::NodeProto) = ProtoBuf.protohash(v)
isequal(v1::NodeProto, v2::NodeProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::NodeProto, v2::NodeProto) = ProtoBuf.protoeq(v1, v2)

mutable struct GraphProto <: ProtoType
    node::Vector{NodeProto}
    name::AbstractString
    initializer::Vector{TensorProto}
    doc_string::AbstractString
    input::Vector{ValueInfoProto}
    output::Vector{ValueInfoProto}
    value_info::Vector{ValueInfoProto}
    GraphProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphProto
const __fnum_GraphProto = Int[1,2,5,10,11,12,13]
meta(t::Type{GraphProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GraphProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::GraphProto) = ProtoBuf.protohash(v)
isequal(v1::GraphProto, v2::GraphProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::GraphProto, v2::GraphProto) = ProtoBuf.protoeq(v1, v2)

mutable struct ModelProto <: ProtoType
    ir_version::Int64
    opset_import::Vector{OperatorSetIdProto}
    producer_name::AbstractString
    producer_version::AbstractString
    domain::AbstractString
    model_version::Int64
    doc_string::AbstractString
    graph::GraphProto
    metadata_props::Vector{StringStringEntryProto}
    ModelProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ModelProto
const __fnum_ModelProto = Int[1,8,2,3,4,5,6,7,14]
meta(t::Type{ModelProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ModelProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ModelProto) = ProtoBuf.protohash(v)
isequal(v1::ModelProto, v2::ModelProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::ModelProto, v2::ModelProto) = ProtoBuf.protoeq(v1, v2)

export Version, AttributeProto_AttributeType, AttributeProto, ValueInfoProto, NodeProto, ModelProto, StringStringEntryProto, GraphProto, TensorProto_DataType, TensorProto_Segment, TensorProto, TensorShapeProto_Dimension, TensorShapeProto, TypeProto_Tensor, TypeProto, OperatorSetIdProto, AttributeProto_AttributeType, AttributeProto

end
