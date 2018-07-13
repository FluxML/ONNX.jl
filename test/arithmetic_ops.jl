using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

#test add:
main_test("$ONNX_TEST_PATH/test_add", read_output("$ONNX_TEST_PATH/test_add"), 
                                      read_input("$ONNX_TEST_PATH/test_add")[1],
                                      read_input("$ONNX_TEST_PATH/test_add")[2])

#test add bcast
main_test("$ONNX_TEST_PATH/test_add_bcast", 
          read_output("$ONNX_TEST_PATH/test_add_bcast"), 
          read_input("$ONNX_TEST_PATH/test_add_bcast")[1],
          read_input("$ONNX_TEST_PATH/test_add_bcast")[2])    

#test mul
main_test("$ONNX_TEST_PATH/test_mul", read_output("$ONNX_TEST_PATH/test_mul"), 
                                      read_input("$ONNX_TEST_PATH/test_mul")[1],
                                      read_input("$ONNX_TEST_PATH/test_mul")[2])

#test mul bcast
main_test("$ONNX_TEST_PATH/test_mul_bcast", read_output("$ONNX_TEST_PATH/test_mul_bcast"), 
                                            read_input("$ONNX_TEST_PATH/test_mul_bcast")[1], 
                                            read_input("$ONNX_TEST_PATH/test_mul_bcast")[2])

#test sub
main_test("$ONNX_TEST_PATH/test_sub", read_output("$ONNX_TEST_PATH/test_sub"), 
                                      read_input("$ONNX_TEST_PATH/test_sub")[1], 
                                      read_input("$ONNX_TEST_PATH/test_sub")[2])

#test sub bcast
main_test("$ONNX_TEST_PATH/test_sub_bcast", read_output("$ONNX_TEST_PATH/test_sub_bcast"), 
                                            read_input("$ONNX_TEST_PATH/test_sub_bcast")[1], 
                                            read_input("$ONNX_TEST_PATH/test_sub_bcast")[2])

#test div
main_test("$ONNX_TEST_PATH/test_div", read_output("$ONNX_TEST_PATH/test_div"), 
                                      read_input("$ONNX_TEST_PATH/test_div")[1], 
                                      read_input("$ONNX_TEST_PATH/test_div")[2])

#test div bcast
main_test("$ONNX_TEST_PATH/test_div_bcast", read_output("$ONNX_TEST_PATH/test_div_bcast"), 
                                            read_input("$ONNX_TEST_PATH/test_div_bcast")[1], 
                                            read_input("$ONNX_TEST_PATH/test_div_bcast")[2])

#test matmul 2d
main_test("$ONNX_TEST_PATH/test_matmul_2d", read_output("$ONNX_TEST_PATH/test_matmul_2d"), 
                                            read_input("$ONNX_TEST_PATH/test_matmul_2d")[1],
                                            read_input("$ONNX_TEST_PATH/test_matmul_2d")[2])
#test exp
main_test("$ONNX_TEST_PATH/test_exp", read_output("$ONNX_TEST_PATH/test_exp"), 
                                      read_input("$ONNX_TEST_PATH/test_exp")[1])

#test reciprocal
main_test("$ONNX_TEST_PATH/test_reciprocal", read_output("$ONNX_TEST_PATH/test_reciprocal"), 
                                             read_input("$ONNX_TEST_PATH/test_reciprocal")[1])

#test floor
main_test("$ONNX_TEST_PATH/test_floor", read_output("$ONNX_TEST_PATH/test_floor"), 
                                        read_input("$ONNX_TEST_PATH/test_floor")[1])

#test ceil
main_test("$ONNX_TEST_PATH/test_ceil", read_output("$ONNX_TEST_PATH/test_ceil"), 
                                       read_input("$ONNX_TEST_PATH/test_ceil")[1])

#test log
main_test("$ONNX_TEST_PATH/test_log", read_output("$ONNX_TEST_PATH/test_log"), 
                                      read_input("$ONNX_TEST_PATH/test_log")[1])

#test pow
main_test("$ONNX_TEST_PATH/test_pow", read_output("$ONNX_TEST_PATH/test_pow"), 
                                      read_input("$ONNX_TEST_PATH/test_pow")[1],
                                      read_input("$ONNX_TEST_PATH/test_pow")[2])

#test pow bcast
main_test("$ONNX_TEST_PATH/test_pow_bcast_array", read_output("$ONNX_TEST_PATH/test_pow_bcast_array"), 
                                            read_input("$ONNX_TEST_PATH/test_pow_bcast_array")[1],
                                            read_input("$ONNX_TEST_PATH/test_pow_bcast_array")[2])

#test relu
main_test("$ONNX_TEST_PATH/test_relu", read_output("$ONNX_TEST_PATH/test_relu"), 
                                       read_input("$ONNX_TEST_PATH/test_relu")[1])

#Test sum one input
main_test("$ONNX_TEST_PATH/test_sum_one_input", 
          read_output("$ONNX_TEST_PATH/test_sum_one_input"), 
          read_input( "$ONNX_TEST_PATH/test_sum_one_input")[1])

#Test sum two inputs
main_test("$ONNX_TEST_PATH/test_sum_two_inputs", 
          read_output("$ONNX_TEST_PATH/test_sum_two_inputs"), 
          read_input("$ONNX_TEST_PATH/test_sum_two_inputs")[1],
          read_input("$ONNX_TEST_PATH/test_sum_two_inputs")[2])


## Trigonometric ops

#Test sin
main_test("$ONNX_TEST_PATH/test_sin", 
          read_output("$ONNX_TEST_PATH/test_sin"), 
          read_input("$ONNX_TEST_PATH/test_sin")[1])

main_test("$ONNX_TEST_PATH/test_sin_example", 
          read_output("$ONNX_TEST_PATH/test_sin_example"), 
          read_input("$ONNX_TEST_PATH/test_sin_example")[1])
#Test cos
main_test("$ONNX_TEST_PATH/test_cos", 
          read_output("$ONNX_TEST_PATH/test_cos"), 
          read_input("$ONNX_TEST_PATH/test_cos")[1])

main_test("$ONNX_TEST_PATH/test_cos_example", 
          read_output("$ONNX_TEST_PATH/test_cos_example"), 
          read_input("$ONNX_TEST_PATH/test_cos_example")[1])

#Test tan
main_test("$ONNX_TEST_PATH/test_tan", 
          read_output("$ONNX_TEST_PATH/test_tan"), 
          read_input("$ONNX_TEST_PATH/test_tan")[1])

main_test("$ONNX_TEST_PATH/test_tan_example", 
          read_output("$ONNX_TEST_PATH/test_tan_example"), 
          read_input("$ONNX_TEST_PATH/test_tan_example")[1])

#test asin
main_test("$ONNX_TEST_PATH/test_asin", 
          read_output("$ONNX_TEST_PATH/test_asin"), 
          read_input("$ONNX_TEST_PATH/test_asin")[1])

main_test("$ONNX_TEST_PATH/test_asin_example", 
          read_output("$ONNX_TEST_PATH/test_asin_example"), 
          read_input("$ONNX_TEST_PATH/test_asin_example")[1])

#test acos
main_test("$ONNX_TEST_PATH/test_acos", 
          read_output("$ONNX_TEST_PATH/test_acos"), 
          read_input("$ONNX_TEST_PATH/test_acos")[1])

main_test("$ONNX_TEST_PATH/test_acos_example", 
          read_output("$ONNX_TEST_PATH/test_acos_example"), 
          read_input("$ONNX_TEST_PATH/test_acos_example")[1])

#test atan
main_test("$ONNX_TEST_PATH/test_atan", 
          read_output("$ONNX_TEST_PATH/test_atan"), 
          read_input("$ONNX_TEST_PATH/test_atan")[1])

main_test("$ONNX_TEST_PATH/test_atan_example", 
          read_output("$ONNX_TEST_PATH/test_atan_example"), 
          read_input("$ONNX_TEST_PATH/test_atan_example")[1])

# Flatten
main_test("$ONNX_TEST_PATH/test_flatten_axis0", read_output("$ONNX_TEST_PATH/test_flatten_axis0"),
             read_input("$ONNX_TEST_PATH/test_flatten_axis0")[1])

# test gemm broadcast
main_test("$ONNX_TEST_PATH/test_gemm_broadcast", read_output("$ONNX_TEST_PATH/test_gemm_broadcast"),
     read_input("$ONNX_TEST_PATH/test_gemm_broadcast")[1], 
     read_input("$ONNX_TEST_PATH/test_gemm_broadcast")[2],
     read_input("$ONNX_TEST_PATH/test_gemm_broadcast")[3])

# test gemm nobroadcast
main_test("$ONNX_TEST_PATH/test_gemm_nobroadcast", read_output("$ONNX_TEST_PATH/test_gemm_nobroadcast"),
     read_input("$ONNX_TEST_PATH/test_gemm_nobroadcast")[1], 
     read_input("$ONNX_TEST_PATH/test_gemm_nobroadcast")[2],
     read_input("$ONNX_TEST_PATH/test_gemm_nobroadcast")[3])

# test unsqueeze
main_test("$ONNX_TEST_PATH/test_unsqueeze", read_output("$ONNX_TEST_PATH/test_unsqueeze"),
     read_input("$ONNX_TEST_PATH/test_unsqueeze")[1])