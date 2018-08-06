using ONNX, Flux, ProtoBuf
using Base.Test

args =  map(x->lowercase(string(x)), ARGS)

name_to_link = Dict()
name_to_link["squeezenet"] = "https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz"
name_to_link["mnist"] = "https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz"
name_to_link["emotion_ferplus"] = "https://www.cntk.ai/OnnxModels/emotion_ferplus/opset_7/emotion_ferplus.tar.gz"
name_to_link["vgg19"] = "https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz"

function read_ip(name)
    ip = readproto(open(name), ONNX.Proto.TensorProto()) |> ONNX.get_array
    if ndims(ip) ==2
        ip = reshape(pi, (size(ip)[1], size(ip)[2], 1, 1))
    elseif ndims(ip==3)
        ip = reshape(ip, (size(ip)[1], size(ip)[2], size(ip)[3], 1))
    end
    return ip
end

function read_ip(name)
    ip = readproto(open(name), ONNX.Proto.TensorProto()) |> ONNX.get_array
    return ip
end

function main(name)
    if !("models" in readdir())
        mkdir("models")
        cd("models")
    else
        cd("models")
    end
    if name in readdir()
        println("Testing predownloaded model")
    else
        run(`wget $(name_to_link[name])`)
        run(`tar -xvzf $name.tar.gz`)
    end
end


main(args[1])   
cd(args[1])
ONNX.load_model("model.onnx")
weights = ONNX.load_weights("weights.bson")
model = include(pwd()*"/model.jl")
num_test=2
if args[1] == "squeezenet"
    num_test = 11
elseif args[1] == "vgg19"
    num_test = 2
end
@testset begin
if (args[1] == "squeezenet") || (args[1] == "vgg19")
    for x=0:num_test
        @test (findmax(model(read_ip("test_data_set_$x/input_0.pb")))[2] == 
            findmax(read_ip("test_data_set_$x/output_0.pb"))[2])
    end
elseif (args[1] == "mnist")
    for x=0:num_test
        @test (findmax(model(read_ip("test_data_set_$x/input_0.pb")))[2] == 
            findmax(read_ip("test_data_set_$x/output_0.pb"))[2])
    end 
else
    for x=0:num_test
        @test (findmax(model(read_ip("test_data_set_$x/input_0.pb")))[2] == 
            findmax(read_ip("test_data_set_$x/output_0.pb"))[2])
    end
end
end