using BSON
using Flux
rawproto(io::IO) = readproto(io, Proto.ModelProto())
rawproto(path::String) = open(rawproto, path)

function maxpool(a::AbstractArray, b, c, d)
    return Flux.maxpool(a, b, pad=c, stride=d)
end

"""
Convert a data type to the corresponding dictionary.
"""
function convert_model(model::Any)
    dict = Dict(f=>get_field(model, f) for f in fieldnames(model))
    return dict
end

"""
Retrieve only the useful information from a AttributeProto
object into a Dict format.
"""
function convert_model(model::Proto.AttributeProto)
    attributes = [:name, :_type, :f, :i, :ints]
    dict = Dict(f=>get_field(model, f) for f in attributes)
    return dict
end

"""
Convert an Array of ValueInfoProto to Array of Dicts.
"""
function convert_model(model::Array{ONNX.Proto.ValueInfoProto,1})
    a = Array{Any, 1}()
    for ele in model
        push!(a, convert_model(ele))
    end
    return a
end

"""
Convert an Array of OperatorSetIdProto to Array of Dicts.
"""
function convert_model(model::Array{ONNX.Proto.OperatorSetIdProto,1})
    a = Array{Any, 1}()
    for ele in model
        push!(a, convert_model(ele))
    end
    return a
end

"""
Convert an Array of StringStringEntryProto to Array of Dicts.
"""
function convert_model(model::Array{ONNX.Proto.StringStringEntryProto,1})
    a = Array{Any, 1}()
    for ele in model
        push!(a, convert_model(ele))
    end
    return a
end

"""
Get the array from a TensorProto object.
"""
function get_array(x::Proto.TensorProto)
  @assert x.data_type == 1 # Float32
  x = reshape(reinterpret(Float32, x.raw_data), x.dims...)
  return permutedims(x, reverse(1:ndims(x)))
end

"""
Convert a ModelProto object to a Model type.
"""
function convert(model::Proto.ModelProto)
    m = Types.Model(model.ir_version,
                convert_model(model.opset_import),
                model.producer_name,
                model.producer_version,
                model.domain, model.model_version,
                model.doc_string, convert(model.graph),
                convert_model(model.metadata_props))
    return m
end

"""
Convert a GraphProto object to Graph type.
"""
function convert(model::Proto.GraphProto)
    temp = model.initializer
    d = Dict()
    for ele in temp
        d[ele.name] = get_array(ele)
    end
    a = Array{Any, 1}()
    for ele in model.node
        push!(a, convert(ele))
    end
    m = Types.Graph(a,
            model.name,
            d, model.doc_string,
            convert_model(model.input),
            convert_model(model.output),
            convert_model(model.value_info))
    return m
end

"""
Convert a Proto.NodeProto to Node type.
"""
function convert(model::Proto.NodeProto)
    m = Types.Node(model.input,
            model.output,
            model.name,
            model.op_type,
            model.domain,
            convert_model(model.attribute),
            model.doc_string)
    return m
end

function parent(path)
    temp = split(path, "/")
    res = ""
    for element in temp
        if (element != temp[end])
            res = res * element * "/"
        end
    end
    return res
end

"""
Read the ONNX model
"""
function read_onnx(model_onnx)::ONNX.Proto.ModelProto
    readproto(open(model_onnx), ONNX.Proto.ModelProto())
end

"""
extract weights from the model
"""
function extract_weights(onnx_proto)
    f = onnx_proto.graph.initializer
    weights = Dict{Symbol, Any}()
    for ele in f
        weights[Symbol(ele.name)] = ONNX.get_array(ele)
    end
    return weights
end

"""
convert weights dict to Dict{String, Any}
"""
#TODO: this seems a bit kludgy, we should probably figure out
# how to get the keys to weights be strings when it's created
function convert_weights(w)
    weights = Dict{String,Any}()
    for elem in keys(w)
        weights[string(elem)] = w[elem]
    end
    return weights
end

"""
Serialize the weights to a binary format and stores in the
weights.bson file.
"""
function write_weights(model_fn, onnx_proto)
    weights = extract_weights(onnx_proto)
    if '/' in model_fn
        cd(parent(model_fn))
    end
    bson("weights.bson", weights)
    return weights
end

"""
Retrieve the dictionary form the binary file (String to Any).
format.
"""
function load_weights_from_bson(name)
    a = BSON.load(name)
    weights = Dict{String, Any}()
    for ele in keys(a)
        weights[string(ele)] = a[ele]
    end
    return weights
end

"""
Create the model.jl file and write the model to it.
"""
function write_julia_file(model, onnx_proto)
    data = ONNX.code(onnx_proto.graph)
    touch("model.jl")
    open("model.jl","w") do file
        write(file, string(data))
    end
    return data
end

"""
Create the two files from the model.onnx file, return weights and model.
"""
function load_model(model)
    onnx_proto = read_onnx(model)
    weights    = write_weights(model, onnx_proto)
    weights    = convert_weights(weights)
    model_expr = write_julia_file(model, onnx_proto)
    return weights, model_expr
end

export maxpool
