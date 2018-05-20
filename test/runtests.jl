using ONNX, Flux, ProtoBuf
using Base.Test

include("ops_tests.jl")

@testset "ONNX" begin

include("constant.jl")
include("logical_ops.jl")
include("pooling.jl")
include("conv.jl")
include("reshape.jl")
include("arithmetic_ops.jl")

end
