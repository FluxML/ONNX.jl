using ProtoBuf
include("onnx_pb.jl")

function convert_model(model::Any)
    '''
    Convert a data type to the corresponding dictionary.
    '''
    dict = Dict(f=>get_field(model, f) for f in fieldnames(model))
    return dict
end

function convert_model(model::AttributeProto)
    '''
    Retrive only the useful information from a AttributeProto
    object  into a Dict format.
    '''
    attributes = [:name, :_type, :f, :i, :ints]
    dict = Dict(f=>get_field(model, f) for f in attributes)
    return dict
end

function get_array(model::TensorProto)
    '''
    Get _the_ array from a TensorProto object.
    Since :raw_data is the only attribute storing valid array data, we\'ll
    retrieve it.
    '''
    return model.raw_data
end
