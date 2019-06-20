using ONNX, Flux, ProtoBuf
include("ops_tests.jl")

# Maxpool 1D default:
ip = read_input("$ONNX_TEST_PATH/test_maxpool_1d_default")
main_test("$ONNX_TEST_PATH/test_maxpool_1d_default", 
          read_output("$ONNX_TEST_PATH/test_maxpool_1d_default"), 
          read_input("$ONNX_TEST_PATH/test_maxpool_1d_default")[1])

# Maxpool 2D default
ip = read_input("$ONNX_TEST_PATH/test_maxpool_2d_default")
main_test("$ONNX_TEST_PATH/test_maxpool_2d_default", 
          read_output("$ONNX_TEST_PATH/test_maxpool_2d_default"),
          read_input("$ONNX_TEST_PATH/test_maxpool_2d_default")[1])

# Maxpool 2D precomputed pads:
ip = read_input("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_pads")
main_test("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_pads", 
          read_output("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_pads"), 
          read_input("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_pads")[1])

# Maxpool 2D precomputed strides:
ip = read_input("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_strides")
main_test("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_strides", 
          read_output("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_strides"), 
          read_input("$ONNX_TEST_PATH/test_maxpool_2d_precomputed_strides")[1])

# Maxpool 2D strides:
ip = read_input("$ONNX_TEST_PATH/test_maxpool_2d_strides")
main_test("$ONNX_TEST_PATH/test_maxpool_2d_strides", 
          read_output("$ONNX_TEST_PATH/test_maxpool_2d_strides"), 
          read_input("$ONNX_TEST_PATH/test_maxpool_2d_strides")[1])

# Averagepool 1D
ip = read_input("$ONNX_TEST_PATH/test_averagepool_1d_default")
    main_test("$ONNX_TEST_PATH/test_averagepool_1d_default", 
        read_output("$ONNX_TEST_PATH/test_averagepool_1d_default"),
            read_input("$ONNX_TEST_PATH/test_averagepool_1d_default"))


# AveragePool 2D Default
ip = read_input("$ONNX_TEST_PATH/test_averagepool_2d_default")
main_test("$ONNX_TEST_PATH/test_averagepool_2d_default", 
            read_output("$ONNX_TEST_PATH/test_averagepool_2d_default"),
                     read_input("$ONNX_TEST_PATH/test_averagepool_2d_default")[1])

# Averagepool 2d pads count include pad                                     
ip = read_input("$ONNX_TEST_PATH/test_averagepool_2d_pads_count_include_pad")
main_test("$ONNX_TEST_PATH/test_averagepool_2d_pads_count_include_pad", 
    read_output("$ONNX_TEST_PATH/test_averagepool_2d_pads_count_include_pad"), 
        read_input("$ONNX_TEST_PATH/test_averagepool_2d_pads_count_include_pad")[1]);

# AveragePool 2d precomputed pads count include pad        
ip = read_input("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_pads_count_include_pad")
main_test("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_pads_count_include_pad", 
    read_output("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_pads_count_include_pad"), 
        read_input("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_pads_count_include_pad")[1])

# Averagepool precomputed strides        
ip = read_input("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_strides")
    main_test("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_strides", 
        read_output("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_strides"),
            read_input("$ONNX_TEST_PATH/test_averagepool_2d_precomputed_strides"))

# AveragePool 2D Strides
ip = read_input("$ONNX_TEST_PATH/test_averagepool_2d_strides")
main_test("$ONNX_TEST_PATH/test_averagepool_2d_strides", 
            read_output("$ONNX_TEST_PATH/test_averagepool_2d_strides"),
                                     read_input("$ONNX_TEST_PATH/test_averagepool_2d_strides")[1])


# Test globalaveragepool
ip = read_input("$ONNX_TEST_PATH/test_globalaveragepool")
main_test("$ONNX_TEST_PATH/test_globalaveragepool", 
    read_output("$ONNX_TEST_PATH/test_globalaveragepool"), 
        read_input("$ONNX_TEST_PATH/test_globalaveragepool")[1])

# Test globalaveragepool precomputed
ip = read_input("$ONNX_TEST_PATH/test_globalaveragepool_precomputed")
main_test("$ONNX_TEST_PATH/test_globalaveragepool_precomputed", 
    read_output("$ONNX_TEST_PATH/test_globalaveragepool_precomputed"), 
        read_input("$ONNX_TEST_PATH/test_globalaveragepool_precomputed")[1])
    
# Test globalmaxpool
ip = read_input("$ONNX_TEST_PATH/test_globalmaxpool")
main_test("$ONNX_TEST_PATH/test_globalmaxpool", 
    read_output("$ONNX_TEST_PATH/test_globalmaxpool"),
        read_input("$ONNX_TEST_PATH/test_globalmaxpool")[1])

# Test globalmaxpool precomputed
ip = read_input("$ONNX_TEST_PATH/test_globalmaxpool_precomputed")
main_test("$ONNX_TEST_PATH/test_globalmaxpool_precomputed", 
    read_output("$ONNX_TEST_PATH/test_globalmaxpool_precomputed"),
        read_input("$ONNX_TEST_PATH/test_globalmaxpool_precomputed")[1])