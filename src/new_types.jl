#=
    Now, we define the new data types.
    Model => ModelProto
    Graph => GraphProto
    Node  => NodeProto
    Attribute => AttributeProto
    The new types will consist of Julian attributes.
    
    The purpose of dealing with these newer type is to make
    the process simpler and easier to debug. 
=# 

module Types

mutable struct ValueInfo
    name::AbstractString
    doc_string::AbstractString
end

mutable struct Node
    input::Vector{AbstractString}
    output::Vector{AbstractString}
    name::AbstractString
    op_type::AbstractString             # Done!
    domain::AbstractString
    attribute::Dict{Any, Any}         # AttributeProto to Dict
    doc_string::AbstractString
end

mutable struct Graph
    node::Array{Any, 1}         
    name::AbstractString
    initializer::Dict{Any ,Any}             #Storing the array data instead of the tensorproto vector.
    doc_string::AbstractString              #in Dict format.
    input::Array{ValueInfo ,1}              # ValueInfoProto -> ValueInfo
    output::Array{ValueInfo, 1}                    # 
    value_info::Array{ValueInfo, 1}                # Done!
end

mutable struct Model
    ir_version::Int64
    opset_import::Array{Any, 1}              #OperatorSetIdProto to Dict
    producer_name::AbstractString
    producer_version::AbstractString            # Done!
    domain::AbstractString
    model_version::Int64
    doc_string::AbstractString
    graph::Graph                  
    metadata_props::Array{Any, 1}            #StringStringEntryProto to Dict
end

export Model, Graph, Node, ValueInfo
end