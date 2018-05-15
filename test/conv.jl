using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# Conv with strides and no pads
ip = read_input("test_conv_with_strides_no_padding")
main_test("test_conv_with_strides_no_padding", read_output("test_conv_with_strides_no_padding"), 
                read_input("test_conv_with_strides_no_padding")[1],
                        read_input("test_conv_with_strides_no_padding")[2])

# Conv with strides and pads
ip = read_input("test_conv_with_strides_padding")
main_test("test_conv_with_strides_padding", read_output("test_conv_with_strides_padding"), 
                read_input("test_conv_with_strides_padding")[1],
                        read_input("test_conv_with_strides_padding")[2])
