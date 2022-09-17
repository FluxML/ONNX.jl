using ONNX
using Test
import Ghost: V
import ONNX.NNlib as NNlib


import ONNXRunTime as OX


path = "test.onnx"

A = [1.0 4.0;
     2.0 5.0;
     3.0 6.0]
B = [7.0 9.0 11.0;
     8.0 10.0 12.0]
model = OX.load_inference(path);
r = model(Dict("x1" => A, "x2" => B))["x3"]

@test r == [58.0 139.0; 64.0 154.0]

# include("ops.jl")
# include("readwrite.jl")
# include("utils.jl")
# include("conversions.jl")
# include("ort.jl")
# include("saveload.jl")
