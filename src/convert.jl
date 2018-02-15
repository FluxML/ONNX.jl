rawproto(io::IO) = readproto(io, Proto.ModelProto())
rawproto(path::String) = open(rawproto, path)

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
Get the array from a TensorProto object.
Since :raw_data is the only attribute storing valid array data, we'll
retrieve it.
"""
function get_array(model::Proto.TensorProto)
    res = Array{Float32, 1}()
    for element in model.raw_data
        push!(res, element)
    end
    return res
end
