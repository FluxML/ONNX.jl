using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# Cast double to float
main_test("$ONNX_TEST_PATH/test_cast_DOUBLE_to_FLOAT", 
            read_output("$ONNX_TEST_PATH/test_cast_DOUBLE_to_FLOAT"), 
                read_input("$ONNX_TEST_PATH/test_cast_DOUBLE_to_FLOAT")[1])

# Cast double to float16
main_test("$ONNX_TEST_PATH/test_cast_DOUBLE_to_FLOAT16", 
            read_output("$ONNX_TEST_PATH/test_cast_DOUBLE_to_FLOAT16"),
                read_input("$ONNX_TEST_PATH/test_cast_DOUBLE_to_FLOAT16")[1])

# Cast Float16 to double
main_test("$ONNX_TEST_PATH/test_cast_FLOAT16_to_DOUBLE", 
            read_output("$ONNX_TEST_PATH/test_cast_FLOAT16_to_DOUBLE"),
                read_input("$ONNX_TEST_PATH/test_cast_FLOAT16_to_DOUBLE")[1])

# Cast Float16 to Float
main_test("$ONNX_TEST_PATH/test_cast_FLOAT16_to_FLOAT", 
            read_output("$ONNX_TEST_PATH/test_cast_FLOAT16_to_FLOAT"),
                read_input("$ONNX_TEST_PATH/test_cast_FLOAT16_to_FLOAT")[1])

# Cast Float to Double
main_test("$ONNX_TEST_PATH/test_cast_FLOAT_to_DOUBLE", 
            read_output("$ONNX_TEST_PATH/test_cast_FLOAT_to_DOUBLE"),
                read_input("$ONNX_TEST_PATH/test_cast_FLOAT_to_DOUBLE")[1])

# Cast Float to Float16
main_test("$ONNX_TEST_PATH/test_cast_FLOAT_to_FLOAT16", 
            read_output("$ONNX_TEST_PATH/test_cast_FLOAT_to_FLOAT16"),
                read_input("$ONNX_TEST_PATH/test_cast_FLOAT_to_FLOAT16")[1])