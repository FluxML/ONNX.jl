using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

#test add:
main_test("test_add", read_output("test_add"), read_input("test_add")[1],
                                                 read_input("test_add")[2])

#test add bcast
main_test("test_add_bcast", read_output("test_add_bcast"), 
                    read_input("test_add_bcast")[1],read_input("test_add_bcast")[2])    

#test mul
main_test("test_mul", read_output("test_mul"), read_input("test_mul")[1],
                                                         read_input("test_mul")[2])

#test mul bcast
main_test("test_mul_bcast", read_output("test_mul_bcast"), 
                        read_input("test_mul_bcast")[1], read_input("test_mul_bcast")[2])

#test sub
main_test("test_sub", read_output("test_sub"), 
                        read_input("test_sub")[1], read_input("test_sub")[2])

#test sub bcast
main_test("test_sub_bcast", read_output("test_sub_bcast"), 
                        read_input("test_sub_bcast")[1], read_input("test_sub_bcast")[2])

#test div
main_test("test_div", read_output("test_div"), 
                                read_input("test_div")[1], read_input("test_div")[2])

#test div bcast
main_test("test_div_bcast", read_output("test_div_bcast"), 
                                read_input("test_div_bcast")[1], read_input("test_div_bcast")[2])

#test matmul 2d
main_test("test_matmul_2d", read_output("test_matmul_2d"), 
                read_input("test_matmul_2d")[1],read_input("test_matmul_2d")[2])
#test exp
main_test("test_exp", read_output("test_exp"), read_input("test_exp")[1])

#test reciprocal
main_test("test_reciprocal", read_output("test_reciprocal"), read_input("test_reciprocal")[1])

#test floor
main_test("test_floor", read_output("test_floor"), read_input("test_floor")[1])

#test ceil
main_test("test_ceil", read_output("test_ceil"), read_input("test_ceil")[1])

#test log
main_test("test_log", read_output("test_log"), read_input("test_log")[1])

#test pow
main_test("test_pow", read_output("test_pow"), read_input("test_pow")[1],read_input("test_pow")[2])

#test pow bcast
main_test("test_pow_bcast", read_output("test_pow_bcast"), 
                read_input("test_pow_bcast")[1],
                        read_input("test_pow_bcast")[2])

#test relu
main_test("test_relu", read_output("test_relu"), read_input("test_relu")[1])

#Test sum one input
main_test("test_sum_one_input", read_output("test_sum_one_input"), 
                                        read_input("test_sum_one_input")[1])

#Test sum two inputs
main_test("test_sum_two_inputs", read_output("test_sum_two_inputs"), 
                read_input("test_sum_two_inputs")[1],read_input("test_sum_two_inputs")[2])


## Trigonometric ops

#Test sin
main_test("test_sin", read_output("test_sin"), read_input("test_sin")[1])
main_test("test_sin_example", read_output("test_sin_example"), read_input("test_sin_example")[1])
#Test cos
main_test("test_cos", read_output("test_cos"), read_input("test_cos")[1])
main_test("test_cos_example", read_output("test_cos_example"), read_input("test_cos_example")[1])

#Test tan
main_test("test_tan", read_output("test_tan"), read_input("test_tan")[1])
main_test("test_tan_example", read_output("test_tan_example"), read_input("test_tan_example")[1])

#test asin
main_test("test_asin", read_output("test_asin"), read_input("test_asin")[1])
main_test("test_asin_example", read_output("test_asin_example"), read_input("test_asin_example")[1])

#test acos
main_test("test_acos", read_output("test_acos"), read_input("test_acos")[1])
main_test("test_acos_example", read_output("test_acos_example"), read_input("test_acos_example")[1])

#test atan
main_test("test_atan", read_output("test_atan"), read_input("test_atan")[1])
main_test("test_atan_example", read_output("test_atan_example"), read_input("test_atan_example")[1])
