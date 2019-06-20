using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

#Test Constant
main_test("$ONNX_TEST_PATH/test_constant", read_output("$ONNX_TEST_PATH/test_constant"))


#Test BatchNorm epsilon
main_test("$ONNX_TEST_PATH/test_batchnorm_epsilon", read_output("$ONNX_TEST_PATH/test_batchnorm_epsilon"), 
    read_input("$ONNX_TEST_PATH/test_batchnorm_epsilon")[1] ,
    read_input("$ONNX_TEST_PATH/test_batchnorm_epsilon")[2],
    read_input("$ONNX_TEST_PATH/test_batchnorm_epsilon")[3], 
    read_input("$ONNX_TEST_PATH/test_batchnorm_epsilon")[4],
    read_input("$ONNX_TEST_PATH/test_batchnorm_epsilon")[5])
"""
# Test BatchNorm example
main_test("$ONNX_TEST_PATH/test_batchnorm_example", 
    read_output("$ONNX_TEST_PATH/test_batchnorm_example"),
    read_input("$ONNX_TEST_PATH/test_batchnorm_example")[1] ,
    read_input("$ONNX_TEST_PATH/test_batchnorm_example")[2],
    read_input("$ONNX_TEST_PATH/test_batchnorm_example")[3], 
    read_input("$ONNX_TEST_PATH/test_batchnorm_example")[4], 
    read_input("$ONNX_TEST_PATH/test_batchnorm_example")[5])
"""