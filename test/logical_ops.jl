using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

#Test and2d
main_test("$ONNX_TEST_PATH/test_and2d", 
          read_output("$ONNX_TEST_PATH/test_and2d"), 
          read_input("$ONNX_TEST_PATH/test_and2d")[1],
          read_input("$ONNX_TEST_PATH/test_and2d")[2])

#Test and3d
main_test("$ONNX_TEST_PATH/test_and3d", 
          read_output("$ONNX_TEST_PATH/test_and3d"),
          read_input("$ONNX_TEST_PATH/test_and3d")[1],
          read_input("$ONNX_TEST_PATH/test_and3d")[2])

#Test and4d
main_test("$ONNX_TEST_PATH/test_and4d", 
          read_output("$ONNX_TEST_PATH/test_and4d"), 
          read_input("$ONNX_TEST_PATH/test_and4d")[1],
          read_input("$ONNX_TEST_PATH/test_and4d")[2])
