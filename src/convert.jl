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
"""
function get_array(x::Proto.TensorProto)
  @assert x.data_type == 1 # Float32
  x = reshape(reinterpret(Float32, x.raw_data), x.dims...)
  return permutedims(x, reverse(1:ndims(x)))
end
