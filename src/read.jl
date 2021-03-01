
"""
    array(p::TensorProto)

Return `p` as an `Array` of the correct type. Second argument can be used to change type of the returned array
"""
function array(p::TensorProto, wrap=Array)
    # Copy pasted from jl
    # Can probably be cleaned up a bit 
    # TODO: Add missing datatypes...
    if p.data_type === TensorProto_DataType.INT64
        if hasproperty(p, :int64_data) && !isempty(p.int64_data)
            return reshape(reinterpret(Int64, p.int64_data), reverse(p.dims)...) |> wrap
        end
        return reshape(reinterpret(Int64, p.raw_data), reverse(p.dims)...) |> wrap
    end

    if p.data_type === TensorProto_DataType.INT32
        if hasproperty(p, :int32_data) && !isempty(p.int32_data)
            return reshape(p.int32_data , reverse(p.dims)...) |> wrap
        end
        return reshape(reinterpret(Int32, p.raw_data), reverse(p.dims)...) |> wrap
    end

    if p.data_type === TensorProto_DataType.INT8
        return reshape(reinterpret(Int8, p.raw_data), reverse(p.dims)...) |> wrap
    end

    if p.data_type === TensorProto_DataType.DOUBLE
        if hasproperty(p, :double_data) && !isempty(p.double_data)
            return reshape(p.double_data , reverse(p.dims)...) |> wrap
        end
        return reshape(reinterpret(Float64, p.raw_data), reverse(p.dims)...) |> wrap
    end

    if p.data_type === TensorProto_DataType.FLOAT
        if hasproperty(p,:float_data) && !isempty(p.float_data)
            return reshape(reinterpret(Float32, p.float_data), reverse(p.dims)...) |> wrap
        end
        return reshape(reinterpret(Float32, p.raw_data), reverse(p.dims)...) |> wrap
    end

    if p.data_type === TensorProto_DataType.FLOAT16
        return reshape(reinterpret(Float16, p.raw_data), reverse(p.dims)...) |> wrap
    end
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
