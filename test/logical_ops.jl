using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

#Test and2d
main_test("test_and2d", read_output("test_and2d"), 
                        read_input("test_and2d")[1],
                                read_input("test_and2d")[2])

#Test and3d
main_test("test_and3d", read_output("test_and3d"),
                read_input("test_and3d")[1],read_input("test_and3d")[2])

#Test and4d
main_test("test_and4d", read_output("test_and4d"), read_input("test_and4d")[1],read_input("test_and4d")[2])