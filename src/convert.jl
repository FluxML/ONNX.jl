"""
    Read the model.pb file into TensorProto object.
"""
function read(name::AbstractString)
    f = open(name)
    return readproto(f, ModelProto())
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
    object  into a Dict format.
"""
function convert_model(model::AttributeProto)
    attributes = [:name, :_type, :f, :i, :ints]
    dict = Dict(f=>get_field(model, f) for f in attributes)
    return dict
end

"""
    Get _the_ array from a TensorProto object.
    Since :raw_data is the only attribute storing valid array data, we\'ll
    retrieve it.
"""
function get_array(model::TensorProto)
    res = Array{Float32, 1}()
    for element in model.raw_data
        push!(res, element)
    end
    return res
end
