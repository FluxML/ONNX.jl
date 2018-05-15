using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

#test reshape one dim:
main_test("test_reshape_one_dim", read_output("test_reshape_extended_dims"),
             read_input("test_reshape_extended_dims")[1],
                read_input("test_reshape_extended_dims")[2])

#test reshape extended dim:
main_test("test_reshape_extended_dims", read_output("test_reshape_extended_dims"),
                 read_input("test_reshape_extended_dims")[1], 
                        read_input("test_reshape_extended_dims")[2])

#test reshape reordered dim:
main_test("test_reshape_reordered_dims", read_output("test_reshape_reordered_dims"), 
                    read_input("test_reshape_reordered_dims")[1], 
                            read_input("test_reshape_reordered_dims")[2])

#test reshape reduced dim
main_test("test_reshape_reduced_dims", read_output("test_reshape_reduced_dims"), 
                    read_input("test_reshape_reduced_dims")[1], 
                        read_input("test_reshape_reduced_dims")[2])
