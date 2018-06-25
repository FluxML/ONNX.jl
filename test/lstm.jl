using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# Test LSTM with no bias
main_test("$ONNX_TEST_PATH/test_lstm_defaults", read_output("$ONNX_TEST_PATH/test_lstm_defaults"), 
    read_input("$ONNX_TEST_PATH/test_lstm_defaults")[1], read_input("$ONNX_TEST_PATH/test_lstm_defaults")[2], 
        read_input("$ONNX_TEST_PATH/test_lstm_defaults")[3])

# Test LSTM with bias
main_test("$ONNX_TEST_PATH/test_lstm_with_initial_bias", read_output("$ONNX_TEST_PATH/test_lstm_with_initial_bias"),
    read_input("$ONNX_TEST_PATH/test_lstm_with_initial_bias")[1], read_input("$ONNX_TEST_PATH/test_lstm_with_initial_bias")[2],
        read_input("$ONNX_TEST_PATH/test_lstm_with_initial_bias")[3], read_input("$ONNX_TEST_PATH/test_lstm_with_initial_bias")[4])