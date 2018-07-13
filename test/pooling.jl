using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# Maxpool 2D pads:
ip = read_input("$ONNX_TEST_PATH/test_maxpool_2d_pads")
main_test("$ONNX_TEST_PATH/test_maxpool_2d_pads", 
          read_output("$ONNX_TEST_PATH/test_maxpool_2d_pads"), 
          read_input("$ONNX_TEST_PATH/test_maxpool_2d_pads")[1])

# Maxpool 2D default
ip = read_input("$ONNX_TEST_PATH/test_maxpool_2d_default")
main_test("$ONNX_TEST_PATH/test_maxpool_2d_default", 
          read_output("$ONNX_TEST_PATH/test_maxpool_2d_default"),
          read_input("$ONNX_TEST_PATH/test_maxpool_2d_default")[1])

# AveragePool 2D Default
ip = read_input("$ONNX_TEST_PATH/test_averagepool_2d_default")
main_test("$ONNX_TEST_PATH/test_averagepool_2d_default", 
            read_output("$ONNX_TEST_PATH/test_averagepool_2d_default"),
                     read_input("$ONNX_TEST_PATH/test_averagepool_2d_default")[1])

# AveragePool 2D Strides
ip = read_input("$ONNX_TEST_PATH/test_averagepool_2d_strides")
main_test("$ONNX_TEST_PATH/test_averagepool_2d_strides", 
            read_output("$ONNX_TEST_PATH/test_averagepool_2d_strides"),
                                     read_input("$ONNX_TEST_PATH/test_averagepool_2d_strides")[1])

# Test globalaveragepool
main_test("$ONNX_TEST_PATH/test_globalaveragepool", 
    read_output("$ONNX_TEST_PATH/test_globalaveragepool"), 
        read_input("$ONNX_TEST_PATH/test_globalaveragepool")[1])

# Test globalmaxpool
main_test("$ONNX_TEST_PATH/test_globalmaxpool", 
    read_output("$ONNX_TEST_PATH/test_globalmaxpool"),
        read_input("$ONNX_TEST_PATH/test_globalmaxpool")[1])