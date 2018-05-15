using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

main_test("test_constant", read_output("test_constant"), )