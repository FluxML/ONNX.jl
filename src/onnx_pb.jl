# syntax: proto2
using ProtoBuf
import ProtoBuf.meta

const Version = (;[
    Symbol("_START_VERSION") => Int32(0),
    Symbol("IR_VERSION_2017_10_10") => Int32(1),
    Symbol("IR_VERSION_2017_10_30") => Int32(2),
    Symbol("IR_VERSION_2017_11_3") => Int32(3),
    Symbol("IR_VERSION_2019_1_22") => Int32(4),
    Symbol("IR_VERSION_2019_3_18") => Int32(5),
    Symbol("IR_VERSION_2019_9_19") => Int32(6),
    Symbol("IR_VERSION") => Int32(7),
]...)

mutable struct StringStringEntryProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function StringStringEntryProto(; kwargs...)
        obj = new(meta(StringStringEntryProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct StringStringEntryProto
const __meta_StringStringEntryProto = Ref{ProtoMeta}()
function meta(::Type{StringStringEntryProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_StringStringEntryProto)
            __meta_StringStringEntryProto[] = target = ProtoMeta(StringStringEntryProto)
            allflds = Pair{Symbol,Union{Type,String}}[:key => AbstractString, :value => AbstractString]
            meta(target, StringStringEntryProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_StringStringEntryProto[]
    end
end
function Base.getproperty(obj::StringStringEntryProto, name::Symbol)
    if name === :key
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :value
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct TensorAnnotation <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorAnnotation(; kwargs...)
        obj = new(meta(TensorAnnotation), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TensorAnnotation
const __meta_TensorAnnotation = Ref{ProtoMeta}()
function meta(::Type{TensorAnnotation})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorAnnotation)
            __meta_TensorAnnotation[] = target = ProtoMeta(TensorAnnotation)
            allflds = Pair{Symbol,Union{Type,String}}[:tensor_name => AbstractString, :quant_parameter_tensor_names => Base.Vector{StringStringEntryProto}]
            meta(target, TensorAnnotation, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TensorAnnotation[]
    end
end
function Base.getproperty(obj::TensorAnnotation, name::Symbol)
    if name === :tensor_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :quant_parameter_tensor_names
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{StringStringEntryProto}
    else
        getfield(obj, name)
    end
end

const TensorProto_DataType = (;[
    Symbol("UNDEFINED") => Int32(0),
    Symbol("FLOAT") => Int32(1),
    Symbol("UINT8") => Int32(2),
    Symbol("INT8") => Int32(3),
    Symbol("UINT16") => Int32(4),
    Symbol("INT16") => Int32(5),
    Symbol("INT32") => Int32(6),
    Symbol("INT64") => Int32(7),
    Symbol("STRING") => Int32(8),
    Symbol("BOOL") => Int32(9),
    Symbol("FLOAT16") => Int32(10),
    Symbol("DOUBLE") => Int32(11),
    Symbol("UINT32") => Int32(12),
    Symbol("UINT64") => Int32(13),
    Symbol("COMPLEX64") => Int32(14),
    Symbol("COMPLEX128") => Int32(15),
    Symbol("BFLOAT16") => Int32(16),
]...)

const TensorProto_DataLocation = (;[
    Symbol("DEFAULT") => Int32(0),
    Symbol("EXTERNAL") => Int32(1),
]...)

mutable struct TensorProto_Segment <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorProto_Segment(; kwargs...)
        obj = new(meta(TensorProto_Segment), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TensorProto_Segment
const __meta_TensorProto_Segment = Ref{ProtoMeta}()
function meta(::Type{TensorProto_Segment})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorProto_Segment)
            __meta_TensorProto_Segment[] = target = ProtoMeta(TensorProto_Segment)
            allflds = Pair{Symbol,Union{Type,String}}[:_begin => Int64, :_end => Int64]
            meta(target, TensorProto_Segment, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TensorProto_Segment[]
    end
end
function Base.getproperty(obj::TensorProto_Segment, name::Symbol)
    if name === :_begin
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :_end
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct TensorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorProto(; kwargs...)
        obj = new(meta(TensorProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TensorProto
const __meta_TensorProto = Ref{ProtoMeta}()
function meta(::Type{TensorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorProto)
            __meta_TensorProto[] = target = ProtoMeta(TensorProto)
            fnum = Int[1,2,3,4,5,6,7,8,12,9,13,14,10,11]
            pack = Symbol[:float_data,:int32_data,:int64_data,:double_data,:uint64_data]
            allflds = Pair{Symbol,Union{Type,String}}[:dims => Base.Vector{Int64}, :data_type => Int32, :segment => TensorProto_Segment, :float_data => Base.Vector{Float32}, :int32_data => Base.Vector{Int32}, :string_data => Base.Vector{Vector{UInt8}}, :int64_data => Base.Vector{Int64}, :name => AbstractString, :doc_string => AbstractString, :raw_data => Vector{UInt8}, :external_data => Base.Vector{StringStringEntryProto}, :data_location => Int32, :double_data => Base.Vector{Float64}, :uint64_data => Base.Vector{UInt64}]
            meta(target, TensorProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TensorProto[]
    end
end
function Base.getproperty(obj::TensorProto, name::Symbol)
    if name === :dims
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :data_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :segment
        return (obj.__protobuf_jl_internal_values[name])::TensorProto_Segment
    elseif name === :float_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float32}
    elseif name === :int32_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :string_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Vector{UInt8}}
    elseif name === :int64_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :doc_string
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :raw_data
        return (obj.__protobuf_jl_internal_values[name])::Vector{UInt8}
    elseif name === :external_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{StringStringEntryProto}
    elseif name === :data_location
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :double_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    elseif name === :uint64_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UInt64}
    else
        getfield(obj, name)
    end
end

mutable struct SparseTensorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function SparseTensorProto(; kwargs...)
        obj = new(meta(SparseTensorProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct SparseTensorProto
const __meta_SparseTensorProto = Ref{ProtoMeta}()
function meta(::Type{SparseTensorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_SparseTensorProto)
            __meta_SparseTensorProto[] = target = ProtoMeta(SparseTensorProto)
            allflds = Pair{Symbol,Union{Type,String}}[:values => TensorProto, :indices => TensorProto, :dims => Base.Vector{Int64}]
            meta(target, SparseTensorProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_SparseTensorProto[]
    end
end
function Base.getproperty(obj::SparseTensorProto, name::Symbol)
    if name === :values
        return (obj.__protobuf_jl_internal_values[name])::TensorProto
    elseif name === :indices
        return (obj.__protobuf_jl_internal_values[name])::TensorProto
    elseif name === :dims
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct TensorShapeProto_Dimension <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorShapeProto_Dimension(; kwargs...)
        obj = new(meta(TensorShapeProto_Dimension), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TensorShapeProto_Dimension
const __meta_TensorShapeProto_Dimension = Ref{ProtoMeta}()
function meta(::Type{TensorShapeProto_Dimension})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorShapeProto_Dimension)
            __meta_TensorShapeProto_Dimension[] = target = ProtoMeta(TensorShapeProto_Dimension)
            allflds = Pair{Symbol,Union{Type,String}}[:dim_value => Int64, :dim_param => AbstractString, :denotation => AbstractString]
            oneofs = Int[1,1,0]
            oneof_names = Symbol[Symbol("value")]
            meta(target, TensorShapeProto_Dimension, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, oneofs, oneof_names)
        end
        __meta_TensorShapeProto_Dimension[]
    end
end
function Base.getproperty(obj::TensorShapeProto_Dimension, name::Symbol)
    if name === :dim_value
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :dim_param
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :denotation
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct TensorShapeProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorShapeProto(; kwargs...)
        obj = new(meta(TensorShapeProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TensorShapeProto
const __meta_TensorShapeProto = Ref{ProtoMeta}()
function meta(::Type{TensorShapeProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorShapeProto)
            __meta_TensorShapeProto[] = target = ProtoMeta(TensorShapeProto)
            allflds = Pair{Symbol,Union{Type,String}}[:dim => Base.Vector{TensorShapeProto_Dimension}]
            meta(target, TensorShapeProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TensorShapeProto[]
    end
end
function Base.getproperty(obj::TensorShapeProto, name::Symbol)
    if name === :dim
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TensorShapeProto_Dimension}
    else
        getfield(obj, name)
    end
end

mutable struct OperatorSetIdProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function OperatorSetIdProto(; kwargs...)
        obj = new(meta(OperatorSetIdProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct OperatorSetIdProto
const __meta_OperatorSetIdProto = Ref{ProtoMeta}()
function meta(::Type{OperatorSetIdProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_OperatorSetIdProto)
            __meta_OperatorSetIdProto[] = target = ProtoMeta(OperatorSetIdProto)
            allflds = Pair{Symbol,Union{Type,String}}[:domain => AbstractString, :version => Int64]
            meta(target, OperatorSetIdProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_OperatorSetIdProto[]
    end
end
function Base.getproperty(obj::OperatorSetIdProto, name::Symbol)
    if name === :domain
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :version
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

const AttributeProto_AttributeType = (;[
    Symbol("UNDEFINED") => Int32(0),
    Symbol("FLOAT") => Int32(1),
    Symbol("INT") => Int32(2),
    Symbol("STRING") => Int32(3),
    Symbol("TENSOR") => Int32(4),
    Symbol("GRAPH") => Int32(5),
    Symbol("SPARSE_TENSOR") => Int32(11),
    Symbol("FLOATS") => Int32(6),
    Symbol("INTS") => Int32(7),
    Symbol("STRINGS") => Int32(8),
    Symbol("TENSORS") => Int32(9),
    Symbol("GRAPHS") => Int32(10),
    Symbol("SPARSE_TENSORS") => Int32(12),
]...)

mutable struct AttributeProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function AttributeProto(; kwargs...)
        obj = new(meta(AttributeProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct AttributeProto (has cyclic type dependency)
const __meta_AttributeProto = Ref{ProtoMeta}()
function meta(::Type{AttributeProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_AttributeProto)
            __meta_AttributeProto[] = target = ProtoMeta(AttributeProto)
            fnum = Int[1,21,13,20,2,3,4,5,6,22,7,8,9,10,11,23]
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :ref_attr_name => AbstractString, :doc_string => AbstractString, :_type => Int32, :f => Float32, :i => Int64, :s => Vector{UInt8}, :t => TensorProto, :g => "GraphProto", :sparse_tensor => SparseTensorProto, :floats => Base.Vector{Float32}, :ints => Base.Vector{Int64}, :strings => Base.Vector{Vector{UInt8}}, :tensors => Base.Vector{TensorProto}, :graphs => "Base.Vector{GraphProto}", :sparse_tensors => Base.Vector{SparseTensorProto}]
            meta(target, AttributeProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_AttributeProto[]
    end
end
function Base.getproperty(obj::AttributeProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :ref_attr_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :doc_string
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :f
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :i
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :s
        return (obj.__protobuf_jl_internal_values[name])::Vector{UInt8}
    elseif name === :t
        return (obj.__protobuf_jl_internal_values[name])::TensorProto
    elseif name === :g
        return (obj.__protobuf_jl_internal_values[name])::Any
    elseif name === :sparse_tensor
        return (obj.__protobuf_jl_internal_values[name])::SparseTensorProto
    elseif name === :floats
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float32}
    elseif name === :ints
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :strings
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Vector{UInt8}}
    elseif name === :tensors
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TensorProto}
    elseif name === :graphs
        return (obj.__protobuf_jl_internal_values[name])::Any
    elseif name === :sparse_tensors
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{SparseTensorProto}
    else
        getfield(obj, name)
    end
end

mutable struct ValueInfoProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ValueInfoProto(; kwargs...)
        obj = new(meta(ValueInfoProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct ValueInfoProto (has cyclic type dependency)
const __meta_ValueInfoProto = Ref{ProtoMeta}()
function meta(::Type{ValueInfoProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ValueInfoProto)
            __meta_ValueInfoProto[] = target = ProtoMeta(ValueInfoProto)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :_type => "TypeProto", :doc_string => AbstractString]
            meta(target, ValueInfoProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ValueInfoProto[]
    end
end
function Base.getproperty(obj::ValueInfoProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :_type
        return (obj.__protobuf_jl_internal_values[name])::Any
    elseif name === :doc_string
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct NodeProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function NodeProto(; kwargs...)
        obj = new(meta(NodeProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct NodeProto (has cyclic type dependency)
const __meta_NodeProto = Ref{ProtoMeta}()
function meta(::Type{NodeProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_NodeProto)
            __meta_NodeProto[] = target = ProtoMeta(NodeProto)
            fnum = Int[1,2,3,4,7,5,6]
            allflds = Pair{Symbol,Union{Type,String}}[:input => Base.Vector{AbstractString}, :output => Base.Vector{AbstractString}, :name => AbstractString, :op_type => AbstractString, :domain => AbstractString, :attribute => Base.Vector{AttributeProto}, :doc_string => AbstractString]
            meta(target, NodeProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_NodeProto[]
    end
end
function Base.getproperty(obj::NodeProto, name::Symbol)
    if name === :input
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    elseif name === :output
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    elseif name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :op_type
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :domain
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :attribute
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AttributeProto}
    elseif name === :doc_string
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct TrainingInfoProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TrainingInfoProto(; kwargs...)
        obj = new(meta(TrainingInfoProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TrainingInfoProto (has cyclic type dependency)
const __meta_TrainingInfoProto = Ref{ProtoMeta}()
function meta(::Type{TrainingInfoProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TrainingInfoProto)
            __meta_TrainingInfoProto[] = target = ProtoMeta(TrainingInfoProto)
            allflds = Pair{Symbol,Union{Type,String}}[:initialization => "GraphProto", :algorithm => "GraphProto", :initialization_binding => Base.Vector{StringStringEntryProto}, :update_binding => Base.Vector{StringStringEntryProto}]
            meta(target, TrainingInfoProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TrainingInfoProto[]
    end
end
function Base.getproperty(obj::TrainingInfoProto, name::Symbol)
    if name === :initialization
        return (obj.__protobuf_jl_internal_values[name])::Any
    elseif name === :algorithm
        return (obj.__protobuf_jl_internal_values[name])::Any
    elseif name === :initialization_binding
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{StringStringEntryProto}
    elseif name === :update_binding
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{StringStringEntryProto}
    else
        getfield(obj, name)
    end
end

mutable struct ModelProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ModelProto(; kwargs...)
        obj = new(meta(ModelProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct ModelProto (has cyclic type dependency)
const __meta_ModelProto = Ref{ProtoMeta}()
function meta(::Type{ModelProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ModelProto)
            __meta_ModelProto[] = target = ProtoMeta(ModelProto)
            fnum = Int[1,8,2,3,4,5,6,7,14,20]
            allflds = Pair{Symbol,Union{Type,String}}[:ir_version => Int64, :opset_import => Base.Vector{OperatorSetIdProto}, :producer_name => AbstractString, :producer_version => AbstractString, :domain => AbstractString, :model_version => Int64, :doc_string => AbstractString, :graph => "GraphProto", :metadata_props => Base.Vector{StringStringEntryProto}, :training_info => Base.Vector{TrainingInfoProto}]
            meta(target, ModelProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ModelProto[]
    end
end
function Base.getproperty(obj::ModelProto, name::Symbol)
    if name === :ir_version
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :opset_import
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{OperatorSetIdProto}
    elseif name === :producer_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :producer_version
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :domain
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :model_version
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :doc_string
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :graph
        return (obj.__protobuf_jl_internal_values[name])::Any
    elseif name === :metadata_props
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{StringStringEntryProto}
    elseif name === :training_info
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TrainingInfoProto}
    else
        getfield(obj, name)
    end
end

mutable struct GraphProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GraphProto(; kwargs...)
        obj = new(meta(GraphProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct GraphProto (has cyclic type dependency)
const __meta_GraphProto = Ref{ProtoMeta}()
function meta(::Type{GraphProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GraphProto)
            __meta_GraphProto[] = target = ProtoMeta(GraphProto)
            fnum = Int[1,2,5,15,10,11,12,13,14]
            allflds = Pair{Symbol,Union{Type,String}}[:node => Base.Vector{NodeProto}, :name => AbstractString, :initializer => Base.Vector{TensorProto}, :sparse_initializer => Base.Vector{SparseTensorProto}, :doc_string => AbstractString, :input => Base.Vector{ValueInfoProto}, :output => Base.Vector{ValueInfoProto}, :value_info => Base.Vector{ValueInfoProto}, :quantization_annotation => Base.Vector{TensorAnnotation}]
            meta(target, GraphProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GraphProto[]
    end
end
function Base.getproperty(obj::GraphProto, name::Symbol)
    if name === :node
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{NodeProto}
    elseif name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :initializer
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TensorProto}
    elseif name === :sparse_initializer
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{SparseTensorProto}
    elseif name === :doc_string
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :input
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ValueInfoProto}
    elseif name === :output
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ValueInfoProto}
    elseif name === :value_info
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ValueInfoProto}
    elseif name === :quantization_annotation
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TensorAnnotation}
    else
        getfield(obj, name)
    end
end

mutable struct TypeProto_Tensor <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TypeProto_Tensor(; kwargs...)
        obj = new(meta(TypeProto_Tensor), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end
const __meta_TypeProto_Tensor = Ref{ProtoMeta}()
function meta(::Type{TypeProto_Tensor})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TypeProto_Tensor)
            __meta_TypeProto_Tensor[] = target = ProtoMeta(TypeProto_Tensor)
            allflds = Pair{Symbol,Union{Type,String}}[:elem_type => Int32, :shape => TensorShapeProto]
            meta(target, TypeProto_Tensor, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TypeProto_Tensor[]
    end
end
function Base.getproperty(obj::TypeProto_Tensor, name::Symbol)
    if name === :elem_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :shape
        return (obj.__protobuf_jl_internal_values[name])::TensorShapeProto
    else
        getfield(obj, name)
    end
end


mutable struct TypeProto_Sequence <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TypeProto_Sequence(; kwargs...)
        obj = new(meta(TypeProto_Sequence), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TypeProto_Sequence (has cyclic type dependency)
const __meta_TypeProto_Sequence = Ref{ProtoMeta}()
function meta(::Type{TypeProto_Sequence})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TypeProto_Sequence)
            __meta_TypeProto_Sequence[] = target = ProtoMeta(TypeProto_Sequence)
            allflds = Pair{Symbol,Union{Type,String}}[:elem_type => "TypeProto"]
            meta(target, TypeProto_Sequence, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TypeProto_Sequence[]
    end
end
function Base.getproperty(obj::TypeProto_Sequence, name::Symbol)
    if name === :elem_type
        return (obj.__protobuf_jl_internal_values[name])::Any
    else
        getfield(obj, name)
    end
end

mutable struct TypeProto_Map <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TypeProto_Map(; kwargs...)
        obj = new(meta(TypeProto_Map), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TypeProto_Map (has cyclic type dependency)
const __meta_TypeProto_Map = Ref{ProtoMeta}()
function meta(::Type{TypeProto_Map})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TypeProto_Map)
            __meta_TypeProto_Map[] = target = ProtoMeta(TypeProto_Map)
            allflds = Pair{Symbol,Union{Type,String}}[:key_type => Int32, :value_type => "TypeProto"]
            meta(target, TypeProto_Map, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TypeProto_Map[]
    end
end
function Base.getproperty(obj::TypeProto_Map, name::Symbol)
    if name === :key_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :value_type
        return (obj.__protobuf_jl_internal_values[name])::Any
    else
        getfield(obj, name)
    end
end

mutable struct TypeProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TypeProto(; kwargs...)
        obj = new(meta(TypeProto), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TypeProto (has cyclic type dependency)
const __meta_TypeProto = Ref{ProtoMeta}()
function meta(::Type{TypeProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TypeProto)
            __meta_TypeProto[] = target = ProtoMeta(TypeProto)
            fnum = Int[1,4,5,6]
            allflds = Pair{Symbol,Union{Type,String}}[:tensor_type => TypeProto_Tensor, :sequence_type => TypeProto_Sequence, :map_type => TypeProto_Map, :denotation => AbstractString]
            oneofs = Int[1,1,1,0]
            oneof_names = Symbol[Symbol("value")]
            meta(target, TypeProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, oneofs, oneof_names)
        end
        __meta_TypeProto[]
    end
end
function Base.getproperty(obj::TypeProto, name::Symbol)
    if name === :tensor_type
        return (obj.__protobuf_jl_internal_values[name])::TypeProto_Tensor
    elseif name === :sequence_type
        return (obj.__protobuf_jl_internal_values[name])::TypeProto_Sequence
    elseif name === :map_type
        return (obj.__protobuf_jl_internal_values[name])::TypeProto_Map
    elseif name === :denotation
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

export Version, AttributeProto_AttributeType, AttributeProto, ValueInfoProto, NodeProto, TrainingInfoProto, ModelProto, StringStringEntryProto, TensorAnnotation, GraphProto, TensorProto_DataType, TensorProto_DataLocation, TensorProto_Segment, TensorProto, SparseTensorProto, TensorShapeProto_Dimension, TensorShapeProto, TypeProto_Tensor, TypeProto_Sequence, TypeProto_Map, TypeProto, OperatorSetIdProto, AttributeProto_AttributeType, AttributeProto, ValueInfoProto, NodeProto, TrainingInfoProto, ModelProto, GraphProto, TypeProto_Sequence, TypeProto_Map, TypeProto
