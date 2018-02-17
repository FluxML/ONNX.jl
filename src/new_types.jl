#=
    Now, we define the new data types.
    Model => ModelProto
    Graph => GraphProto
    Node  => NodeProto
    Attribute => AttributeProto
    The new types will consist of Julian attributes.
=# 

module Types

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
    initializer::Dict{Any, Any}             #Storing the array data instead of the tensorproto vector.
    doc_string::AbstractString              #in Dict format.
    input::Array{Any, 1}                     #
    output::Array{Any, 1}                    # ValueInfoProto to Dict type.
    value_info::Array{Any, 1}                #
end

mutable struct Model
    ir_version::Int64
    opset_import::Array{Any, 1}              #OperatorSetIdProto to Dict
    producer_name::AbstractString
    producer_version::AbstractString
    domain::AbstractString
    model_version::Int64
    doc_string::AbstractString
    graph::Graph                  
    metadata_props::Array{Any, 1}            #StringStringEntryProto to Dict
end

export Model, Graph, Node
end