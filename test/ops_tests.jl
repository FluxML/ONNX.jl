using ONNX, Flux, ProtoBuf
using DataFlow: Call, vertex, syntax, constant
using Base.Test

# test taken from : https://github.com/onnx/onnx/tree/master/onnx/backend/test/data 

function read_input(folder_name)
    ar = Array{Any, 1}()
    for ele in readdir(folder_name*"/test_data_set_0")
        push!(ar, readproto(open(folder_name*"/test_data_set_0/"*ele), ONNX.Proto.TensorProto()) |> ONNX.get_array)
    end
    return ar[1:end-1]
end

function read_output(folder_name)
    ar = Array{Any, 1}()
    for ele in readdir(folder_name*"/test_data_set_0")
        push!(ar, readproto(open(folder_name*"/test_data_set_0/"*ele), ONNX.Proto.TensorProto()) |> ONNX.get_array)
    end
    return ar[end]
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

function test(filename)
    @test ONNX.ops[Symbol(get_optype(read_model(filename)))](get_dict(read_model(filename)), 
                    constant(read_input(filename)[1]), 
                        constant(permutedims(read_data(filename)[2], reverse(1:ndims(read_data(filename)[2]))))) |> syntax |> eval== read_output(filename)
end