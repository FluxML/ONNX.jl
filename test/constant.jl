using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

main_test("$ONNX_TEST_PATH/test_constant", read_output("$ONNX_TEST_PATH/test_constant"), )
