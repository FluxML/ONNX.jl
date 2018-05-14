using ONNX
using Base.Test

include("ops_tests.jl")

#Tests:

#test add:
main_test("test_add", read_output("test_add"), read_input("test_add")[1], read_input("test_add")[2])

#test add bcast
main_test("test_add_bcast", read_output("test_add_bcast"), 
                    read_input("test_add_bcast")[1],read_input("test_add_bcast")[2])    

#test constant:
main_test("test_constant", read_output("test_constant"), )

#test matmul 2d
main_test("test_matmul_2d", read_output("test_matmul_2d"), 
                read_input("test_matmul_2d")[1],read_input("test_matmul_2d")[2])

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