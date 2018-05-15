using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# Maxpool 2D pads:
ip = read_input("test_maxpool_2d_pads")
main_test("test_maxpool_2d_pads", read_output("test_maxpool_2d_pads"), read_input("test_maxpool_2d_pads")[1])

# Maxpool 2D default
ip = read_input("test_maxpool_2d_default")
main_test("test_maxpool_2d_default", read_output("test_maxpool_2d_default"),
                                         read_input("test_maxpool_2d_default")[1])