using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# test basic conv with padding
ip = read_input("$ONNX_TEST_PATH/test_basic_conv_with_padding")
main_test("$ONNX_TEST_PATH/test_basic_conv_with_padding",
    read_output("$ONNX_TEST_PATH/test_basic_conv_with_padding"),
        read_input("$ONNX_TEST_PATH/test_basic_conv_with_padding")[1], 
            read_input("$ONNX_TEST_PATH/test_basic_conv_with_padding")[2])

# test basic conv without padding
ip = read_input("$ONNX_TEST_PATH/test_basic_conv_without_padding")
main_test("$ONNX_TEST_PATH/test_basic_conv_without_padding", 
    read_output("$ONNX_TEST_PATH/test_basic_conv_without_padding"),
        read_input("$ONNX_TEST_PATH/test_basic_conv_without_padding")[1], 
            read_input("$ONNX_TEST_PATH/test_basic_conv_without_padding")[2])

# Conv with strides and no pads
ip = read_input("$ONNX_TEST_PATH/test_conv_with_strides_no_padding")
main_test("$ONNX_TEST_PATH/test_conv_with_strides_no_padding", 
          read_output("$ONNX_TEST_PATH/test_conv_with_strides_no_padding"), 
          read_input("$ONNX_TEST_PATH/test_conv_with_strides_no_padding")[1],
          read_input("$ONNX_TEST_PATH/test_conv_with_strides_no_padding")[2])

# Conv with strides and pads
ip = read_input("$ONNX_TEST_PATH/test_conv_with_strides_padding")
main_test("$ONNX_TEST_PATH/test_conv_with_strides_padding", 
          read_output("$ONNX_TEST_PATH/test_conv_with_strides_padding"), 
          read_input("$ONNX_TEST_PATH/test_conv_with_strides_padding")[1],
          read_input("$ONNX_TEST_PATH/test_conv_with_strides_padding")[2])

# Test Dropout default          
main_test("$ONNX_TEST_PATH/test_dropout_default", 
            read_output("$ONNX_TEST_PATH/test_dropout_default"),
            read_input("$ONNX_TEST_PATH/test_dropout_default")[1])

# Test Dropout random          
main_test("$ONNX_TEST_PATH/test_dropout_random", 
            read_output("$ONNX_TEST_PATH/test_dropout_random"),
            read_input("$ONNX_TEST_PATH/test_dropout_random")[1])