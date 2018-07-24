using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# test and 3v1d
main_test("$ONNX_TEST_PATH/test_and_bcast3v1d", 
    read_output("$ONNX_TEST_PATH/test_and_bcast3v1d"),
        read_input("$ONNX_TEST_PATH/test_and_bcast3v1d")[1], 
            read_input("$ONNX_TEST_PATH/test_and_bcast3v1d")[2])

# test and 3v2d
main_test("$ONNX_TEST_PATH/test_and_bcast3v2d", 
    read_output("$ONNX_TEST_PATH/test_and_bcast3v2d"),
        read_input("$ONNX_TEST_PATH/test_and_bcast3v2d")[1], 
            read_input("$ONNX_TEST_PATH/test_and_bcast3v2d")[2])

# test and 4v2d
main_test("$ONNX_TEST_PATH/test_and_bcast4v2d", 
    read_output("$ONNX_TEST_PATH/test_and_bcast4v2d"),
        read_input("$ONNX_TEST_PATH/test_and_bcast4v2d")[1], 
            read_input("$ONNX_TEST_PATH/test_and_bcast4v2d")[2])

# test and 4v3d
main_test("$ONNX_TEST_PATH/test_and_bcast4v3d", 
    read_output("$ONNX_TEST_PATH/test_and_bcast4v3d"),
        read_input("$ONNX_TEST_PATH/test_and_bcast4v3d")[1], 
            read_input("$ONNX_TEST_PATH/test_and_bcast4v3d")[2])

# test and 4v4d
main_test("$ONNX_TEST_PATH/test_and_bcast4v4d", 
    read_output("$ONNX_TEST_PATH/test_and_bcast4v4d"),
        read_input("$ONNX_TEST_PATH/test_and_bcast4v4d")[1], 
            read_input("$ONNX_TEST_PATH/test_and_bcast4v4d")[2])

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

# Test identity
main_test("$ONNX_TEST_PATH/test_identity", 
    read_output("$ONNX_TEST_PATH/test_identity"),
        read_input("$ONNX_TEST_PATH/test_identity")[1])

# Test equal
main_test("$ONNX_TEST_PATH/test_equal", 
    read_output("$ONNX_TEST_PATH/test_equal"),
        read_input("$ONNX_TEST_PATH/test_equal")[1],
            read_input("$ONNX_TEST_PATH/test_equal")[2])

# Test equal bcast
main_test("$ONNX_TEST_PATH/test_equal_bcast", 
    read_output("$ONNX_TEST_PATH/test_equal_bcast"),
        read_input("$ONNX_TEST_PATH/test_equal_bcast")[1],
            read_input("$ONNX_TEST_PATH/test_equal_bcast")[2])

# Test greater
main_test("$ONNX_TEST_PATH/test_greater", 
    read_output("$ONNX_TEST_PATH/test_greater"),
        read_input("$ONNX_TEST_PATH/test_greater")[1],
            read_input("$ONNX_TEST_PATH/test_greater")[2])

# Test greater bcast
main_test("$ONNX_TEST_PATH/test_greater_bcast", 
    read_output("$ONNX_TEST_PATH/test_greater_bcast"),
        read_input("$ONNX_TEST_PATH/test_greater_bcast")[1],
            read_input("$ONNX_TEST_PATH/test_greater_bcast")[2])

# test xor2d
main_test("$ONNX_TEST_PATH/test_xor2d", 
    read_output("$ONNX_TEST_PATH/test_xor2d"),
        read_input("$ONNX_TEST_PATH/test_xor2d")[1],
            read_input("$ONNX_TEST_PATH/test_xor2d")[2])

# test xor3d
main_test("$ONNX_TEST_PATH/test_xor3d", 
    read_output("$ONNX_TEST_PATH/test_xor3d"),
        read_input("$ONNX_TEST_PATH/test_xor3d")[1],
            read_input("$ONNX_TEST_PATH/test_xor3d")[2])

# test xor4d
main_test("$ONNX_TEST_PATH/test_xor4d", 
    read_output("$ONNX_TEST_PATH/test_xor4d"),
        read_input("$ONNX_TEST_PATH/test_xor4d")[1],
            read_input("$ONNX_TEST_PATH/test_xor4d")[2])

# test xor bcast 3v1d
main_test("$ONNX_TEST_PATH/test_xor_bcast3v1d", 
    read_output("$ONNX_TEST_PATH/test_xor_bcast3v1d"),
        read_input("$ONNX_TEST_PATH/test_xor_bcast3v1d")[1],
            read_input("$ONNX_TEST_PATH/test_xor_bcast3v1d")[2])

# test xor bcast 3v2d
main_test("$ONNX_TEST_PATH/test_xor_bcast3v2d", 
    read_output("$ONNX_TEST_PATH/test_xor_bcast3v2d"),
        read_input("$ONNX_TEST_PATH/test_xor_bcast3v2d")[1],
            read_input("$ONNX_TEST_PATH/test_xor_bcast3v2d")[2])

# test xor bcast 4v2d
main_test("$ONNX_TEST_PATH/test_xor_bcast4v2d", 
    read_output("$ONNX_TEST_PATH/test_xor_bcast4v2d"),
        read_input("$ONNX_TEST_PATH/test_xor_bcast4v2d")[1],
            read_input("$ONNX_TEST_PATH/test_xor_bcast4v2d")[2])

# test xor bcast 4v3d
main_test("$ONNX_TEST_PATH/test_xor_bcast4v3d", 
    read_output("$ONNX_TEST_PATH/test_xor_bcast4v3d"),
        read_input("$ONNX_TEST_PATH/test_xor_bcast4v3d")[1],
            read_input("$ONNX_TEST_PATH/test_xor_bcast4v3d")[2])

# test xor bcast 4v4d
main_test("$ONNX_TEST_PATH/test_xor_bcast4v4d", 
    read_output("$ONNX_TEST_PATH/test_xor_bcast4v4d"),
        read_input("$ONNX_TEST_PATH/test_xor_bcast4v4d")[1],
            read_input("$ONNX_TEST_PATH/test_xor_bcast4v4d")[2])