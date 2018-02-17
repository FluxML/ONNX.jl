#=
    Now, we define the new data types.
    Model => ModelProto
    Graph => GraphProto
    Node  => NodeProto
    Attribute => AttributeProto
    The new types will consist of Julian attributes.
=# 

mutable struct Node
    input::Vector{AbstractString}
    output::Vector{AbstractString}
    name::AbstractString
    op_type::AbstractString
    domain::AbstractString
    attribute::Dict{Symbol, Any}            #AttributeProto to Dict
    doc_string::AbstractString
end

mutable struct Graph
    node::Vector{Node}         
    name::AbstractString
    initializer::Dict{Any, Array{Any, 1}}   #Storing the array data instead of the tensorproto vector.
    doc_string::AbstractString              #in Dict format.
    input::Vector{Dict}                     #
    output::Vector{Dict}                    # ValueInfoProto to Dict type.
    value_info::Vector{Dict}                #
end

mutable struct Model
    ir_version::Int64
    opset_import::Vector{Dict}              #OperatorSetIdProto to Dict
    producer_name::AbstractString
    producer_version::AbstractString
    domain::AbstractString
    model_version::Int64
    doc_string::AbstractString
    graph::Graph                  
    metadata_props::Vector{Dict}            #StringStringEntryProto to Dict
end
