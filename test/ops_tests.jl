using ONNX, Flux, ProtoBuf
using DataFlow: Call, vertex, syntax, constant
using Test
using Base:run
# test taken from : https://github.com/onnx/onnx/tree/master/onnx/backend/test/data 
# clone onnx here if onnx dir does not exist

if !("onnx" in readdir())
    # clone the package here
    println("Downloading test data....")
    Base.run(`git clone https://github.com/onnx/onnx.git`)
end

ONNX_PATH = "./onnx"

ONNX_TEST_PATH = "$ONNX_PATH/onnx/backend/test/data/node"


function read_input(folder_name)
    ar = Array{Any, 1}()
    for ele in readdir(folder_name*"/test_data_set_0")
        push!(ar, Float32.(readproto(open(folder_name*"/test_data_set_0/"*ele), ONNX.Proto.TensorProto()) |> ONNX.get_array))
    end
    return ar[1:end-1]
end

function read_output(folder_name)
    ar = Array{Any, 1}()
    for ele in readdir(folder_name*"/test_data_set_0")
        push!(ar, Float32.(readproto(open(folder_name*"/test_data_set_0/"*ele), ONNX.Proto.TensorProto()) |> ONNX.get_array))
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

function main_test(filename,op_expected, ip...)
        if Symbol(get_optype(read_model(filename))) == :Constant
            @test get_dict(read_model(filename))[:value] |> ONNX.get_array == op_expected
        
        elseif Symbol(get_optype(read_model(filename))) == :Conv
        
            temp = ONNX.ops[Symbol(get_optype(read_model(filename)))](get_dict(read_model(filename)),
                                                                                     Symbol("ip[1]"), Symbol("ip[2]")) |> syntax
            
            touch("temp_conv.jl")
            open("temp_conv.jl","w") do file
                str1 = "flipkernel(x) = x[end:-1:1, end:-1:1, :, :] \n"                     # Remove when Flux directly supports it
                write(file, str1*string(temp))
            end
            model = include("temp_conv.jl")
            rm("temp_conv.jl")
            @test model == op_expected
        elseif Symbol(get_optype(read_model(filename))) == :MaxPool
            temp = ONNX.ops[Symbol(get_optype(read_model(filename)))](get_dict(read_model(filename)),
                                                                                     Symbol("ip[1]")) |> syntax
            touch("temp_maxpool.jl")
            open("temp_maxpool.jl","w") do file
                write(file, string(temp))
            end
            
            model = include("temp_maxpool.jl")
            rm("temp_maxpool.jl")
            @test model == op_expected
        elseif Symbol(get_optype(read_model(filename))) == :AveragePool
            temp = ONNX.ops[Symbol(get_optype(read_model(filename)))](get_dict(read_model(filename)),
                                                                                     Symbol("ip[1]")) |> syntax
            touch("temp_averagepool.jl")
            open("temp_averagepool.jl","w") do file
                write(file, string(temp))
            end
            
            model = include("temp_averagepool.jl")
            rm("temp_averagepool.jl")
            @test model ≈ op_expected atol=0.001
        elseif Symbol(get_optype(read_model(filename))) in [:GlobalAveragePool, :GlobalMaxPool]
            temp = ONNX.ops[Symbol(get_optype(read_model(filename)))](get_dict(read_model(filename)),
                                                                                     Symbol("ip[1]")) |> syntax
            touch("temp_averagepool.jl")
            open("temp_averagepool.jl","w") do file
                write(file, "using Statistics \n" * string(temp))
            end
            model = include("temp_averagepool.jl")
            rm("temp_averagepool.jl")
            @test model ≈ op_expected atol=0.001
        elseif Symbol(get_optype(read_model(filename))) in [:Expand, :Concat]
            temp = ONNX.ops[Symbol(get_optype(read_model(filename)))](get_dict(read_model(filename)),
                                                                                     Symbol("ip[1]"), Symbol("ip[2]")) |> syntax
            touch("temp_expand.jl")
            open("temp_expand.jl","w") do file
                write(file, string(temp))
            end
            
            model = include("temp_expand.jl")
            rm("temp_expand.jl")
            @test model ≈ op_expected atol=0.001
        else
                @test ONNX.ops[Symbol(get_optype(read_model(filename)))](get_dict(read_model(filename)),
                                 ip...) |> syntax |> eval ≈ op_expected atol=0.001
        end
end
