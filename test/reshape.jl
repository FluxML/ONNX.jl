using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

#test reshape one dim:
main_test("$ONNX_TEST_PATH/test_reshape_one_dim", 
          read_output("$ONNX_TEST_PATH/test_reshape_extended_dims"),
          read_input("$ONNX_TEST_PATH/test_reshape_extended_dims")[1],
          read_input("$ONNX_TEST_PATH/test_reshape_extended_dims")[2])

#test reshape extended dim:
main_test("$ONNX_TEST_PATH/test_reshape_extended_dims", 
          read_output("$ONNX_TEST_PATH/test_reshape_extended_dims"),
          read_input("$ONNX_TEST_PATH/test_reshape_extended_dims")[1], 
          read_input("$ONNX_TEST_PATH/test_reshape_extended_dims")[2])

#test reshape reordered dim:
main_test("$ONNX_TEST_PATH/test_reshape_reordered_dims", 
          read_output("$ONNX_TEST_PATH/test_reshape_reordered_dims"), 
          read_input("$ONNX_TEST_PATH/test_reshape_reordered_dims")[1], 
          read_input("$ONNX_TEST_PATH/test_reshape_reordered_dims")[2])

#test reshape reduced dim
main_test("$ONNX_TEST_PATH/test_reshape_reduced_dims", 
          read_output("$ONNX_TEST_PATH/test_reshape_reduced_dims"), 
          read_input("$ONNX_TEST_PATH/test_reshape_reduced_dims")[1], 
          read_input("$ONNX_TEST_PATH/test_reshape_reduced_dims")[2])

## Transpose test:

main_test("$ONNX_TEST_PATH/test_transpose_all_permutations_0", 
          read_output("$ONNX_TEST_PATH/test_transpose_all_permutations_0"), 
          read_input("$ONNX_TEST_PATH/test_transpose_all_permutations_0")[1])

main_test("$ONNX_TEST_PATH/test_transpose_all_permutations_1", 
          read_output("$ONNX_TEST_PATH/test_transpose_all_permutations_1"), 
          read_input("$ONNX_TEST_PATH/test_transpose_all_permutations_1")[1])

main_test("$ONNX_TEST_PATH/test_transpose_all_permutations_2", 
          read_output("$ONNX_TEST_PATH/test_transpose_all_permutations_2"), 
          read_input("$ONNX_TEST_PATH/test_transpose_all_permutations_2")[1])
                    
main_test("$ONNX_TEST_PATH/test_transpose_all_permutations_3", 
          read_output("$ONNX_TEST_PATH/test_transpose_all_permutations_3"), 
          read_input("$ONNX_TEST_PATH/test_transpose_all_permutations_3")[1])

main_test("$ONNX_TEST_PATH/test_transpose_all_permutations_4", 
          read_output("$ONNX_TEST_PATH/test_transpose_all_permutations_4"), 
          read_input("$ONNX_TEST_PATH/test_transpose_all_permutations_4")[1])

main_test("$ONNX_TEST_PATH/test_transpose_all_permutations_5", 
          read_output("$ONNX_TEST_PATH/test_transpose_all_permutations_5"), 
          read_input("$ONNX_TEST_PATH/test_transpose_all_permutations_5")[1])

## Test concat

#Test concat 1d axis 0
#ip = read_input("$ONNX_TEST_PATH/test_concat_1d_axis_0")
#main_test("$ONNX_TEST_PATH/test_concat_1d_axis_0", 
#   read_output("$ONNX_TEST_PATH/test_concat_1d_axis_0"), 
#        read_input("$ONNX_TEST_PATH/test_concat_1d_axis_0")[1],
#            read_input("$ONNX_TEST_PATH/test_concat_1d_axis_0")[2])

#Test concat 2d axis 0
#ip = read_input("$ONNX_TEST_PATH/test_concat_2d_axis_0")
#main_test("$ONNX_TEST_PATH/test_concat_2d_axis_0", 
#    read_output("$ONNX_TEST_PATH/test_concat_2d_axis_0"), 
#        read_input("$ONNX_TEST_PATH/test_concat_2d_axis_0")[1],
#            read_input("$ONNX_TEST_PATH/test_concat_2d_axis_0")[2])

#Test concat 2d axis 1
#ip = read_input("$ONNX_TEST_PATH/test_concat_2d_axis_1")
#main_test("$ONNX_TEST_PATH/test_concat_2d_axis_1", 
#    read_output("$ONNX_TEST_PATH/test_concat_2d_axis_1"), 
#        read_input("$ONNX_TEST_PATH/test_concat_2d_axis_1")[1],
#            read_input("$ONNX_TEST_PATH/test_concat_2d_axis_1")[2])

#Test concat 3d axis 0
#ip = read_input("$ONNX_TEST_PATH/test_concat_3d_axis_0")
#main_test("$ONNX_TEST_PATH/test_concat_3d_axis_0", 
#    read_output("$ONNX_TEST_PATH/test_concat_3d_axis_0"), 
#        read_input("$ONNX_TEST_PATH/test_concat_3d_axis_0")[1],
#            read_input("$ONNX_TEST_PATH/test_concat_3d_axis_0")[2])

#Test concat 3d axis 1
#ip = read_input("$ONNX_TEST_PATH/test_concat_3d_axis_1")
#main_test("$ONNX_TEST_PATH/test_concat_3d_axis_1", 
#    read_output("$ONNX_TEST_PATH/test_concat_3d_axis_1"), 
#        read_input("$ONNX_TEST_PATH/test_concat_3d_axis_1")[1],
#            read_input("$ONNX_TEST_PATH/test_concat_3d_axis_1")[2])

#Test concat 3d axis 2
#ip = read_input("$ONNX_TEST_PATH/test_concat_3d_axis_2")
#main_test("$ONNX_TEST_PATH/test_concat_3d_axis_2", 
#    read_output("$ONNX_TEST_PATH/test_concat_3d_axis_2"), 
#        read_input("$ONNX_TEST_PATH/test_concat_3d_axis_2")[1],
#            read_input("$ONNX_TEST_PATH/test_concat_3d_axis_2")[2])
