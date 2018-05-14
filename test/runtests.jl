using ONNX
using Base.Test

include("ops_tests.jl")

#Tests:

## Arithmetic Operator Tests: Add, Sub, Mul, Div

#test add:
main_test("test_add", read_output("test_add"), read_input("test_add")[1], read_input("test_add")[2])

#test add bcast
main_test("test_add_bcast", read_output("test_add_bcast"), 
                    read_input("test_add_bcast")[1],read_input("test_add_bcast")[2])    

#test mul
main_test("test_mul", read_output("test_mul"), read_input("test_mul")[1], read_input("test_mul")[2])

#test mul bcast
main_test("test_mul_bcast", read_output("test_mul_bcast"), read_input("test_mul_bcast")[1], read_input("test_mul_bcast")[2])

#test sub
main_test("test_sub", read_output("test_sub"), read_input("test_sub")[1], read_input("test_sub")[2])

#test sub bcast
main_test("test_sub_bcast", read_output("test_sub_bcast"), read_input("test_sub_bcast")[1], read_input("test_sub_bcast")[2])

#test div
main_test("test_div", read_output("test_div"), read_input("test_div")[1], read_input("test_div")[2])

#test div bcast
main_test("test_div_bcast", read_output("test_div_bcast"), read_input("test_div_bcast")[1], read_input("test_div_bcast")[2])

## Constant Test
#test constant:
main_test("test_constant", read_output("test_constant"), )

#test matmul 2d
main_test("test_matmul_2d", read_output("test_matmul_2d"), 
                read_input("test_matmul_2d")[1],read_input("test_matmul_2d")[2])

## Reshape Tests

#test reshape one dim:
main_test("test_reshape_one_dim", read_output("test_reshape_extended_dims"),
             read_input("test_reshape_extended_dims")[1],read_input("test_reshape_extended_dims")[2])

#test reshape extended dim:
main_test("test_reshape_extended_dims", read_output("test_reshape_extended_dims"),
                 read_input("test_reshape_extended_dims")[1], read_input("test_reshape_extended_dims")[2])

#test reshape reordered dim:
main_test("test_reshape_reordered_dims", read_output("test_reshape_reordered_dims"), 
                    read_input("test_reshape_reordered_dims")[1], 
                            read_input("test_reshape_reordered_dims")[2])

#test reshape reduced dim
main_test("test_reshape_reduced_dims", read_output("test_reshape_reduced_dims"), 
                    read_input("test_reshape_reduced_dims")[1], read_input("test_reshape_reduced_dims")[2])

## Activation Tests

#test relu
main_test("test_relu", read_output("test_relu"), read_input("test_relu")[1])

## Misc tests

#test exp
main_test("test_exp", read_output("test_exp"), read_input("test_exp")[1])


