using ONNX
using Base.Test
using ONNX, Flux, ProtoBuf
using DataFlow: Call, vertex, syntax, constant

# write your own tests here

function read_data(folder_name)
    ar = Array{Any, 1}()
    for ele in readdir(folder_name*"/test_data_set_0")
        push!(ar, readproto(open(folder_name*"/test_data_set_0/"*ele), ONNX.Proto.TensorProto()) |> ONNX.get_array)
    end
    return ar[1:end-1]
end

function read_model(folder_name)
    for ele in readdir(folder_name)
        if ele=="model.onnx"
            return readproto(open(folder_name*"/model.onnx"), ONNX.Proto.ModelProto())
        end
    end
end

function get_optype(a::ONNX.Proto.ModelProto)
    g = ONNX.convert(a.graph)
    return g.node[1].op_type
end

function get_dict(a::ONNX.Proto.ModelProto)
    g = ONNX.convert(a.graph)
    return g.node[1].attribute
end


