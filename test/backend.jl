const ONNX_RELEASE_URL = "https://github.com/ordicker/ONNXBackendTests.jl/releases/download/v1.13.1/backend_data.tar" # TODO: find a better solution

@testset "Backend" begin
    import ONNX.ProtoBuf: encode, decode, ProtoEncoder, ProtoDecoder
    import ONNX: load, array
    import Tar: extract
    import Umlaut: Tape, play!
    import Downloads: download

    onnx_release_path = dirname(@__DIR__) * "/test/backend_tests/"
    if !isdir(onnx_release_path)
        mkpath(onnx_release_path)
        onnx_release_tar_path = download(ONNX_RELEASE_URL)
        extract(onnx_release_tar_path, onnx_release_path)
        rm(onnx_release_tar_path)
    end

    "filename.pb to julia array"
    function pb_to_array(filename::String)
        pb = decode(ProtoDecoder(open(filename)), TensorProto)
        return array(pb)
    end

    """
        outputs(dirname)

    load all the outputs in the testing folder.
    directory structure is:
    dir (test_<name>)
    |--model.onnx
    |--test_data_set_0
       |-- input_0.pb
       |-- ...
       |-- input_N.pb
       |-- output_0.pb
       |-- ...
       |-- output_N.pb
    load all output_X.pb and convert to Julia array
    """
    function outputs(dirname::String)
        readdir(dirname*"/test_data_set_0",join=true)|>
            f->filter(contains(r"\/output.*\.pb"),f).|>
            pb_to_array
    end
    """
        eval_model(dirname)

    run onnx backend model using the data in directory.
    directory structure is:
    dir (test_<name>)
    |--model.onnx
    |--test_data_set_0
       |-- input_0.bp
       |-- ...
       |-- input_N.bp
       |-- output_0.bp
    Loads model.onnx, input_X.pb using ONNX.jl, and run it.
    """
    function load_model_and_inputs(dirname::String)
        ## load inputs
        inputs = readdir(dirname*"/test_data_set_0",join=true)|>
            f->filter(contains(r"\/input.*\.pb"),f).|>
            pb_to_array
        ## load the model
        model = load(dirname*"/model.onnx",inputs...)
        ## run it
        return model, inputs
    end

    @testset "Nodes" begin
        prefix = onnx_release_path * "/data/node/"
        #for dir in readdir(prefix) # TODO: pass all the tests :)
        for dirname in ["test_add",
                        "test_min_example",
                        ##"test_min_float16",
                        "test_min_float32",
                        "test_min_float64",
                        #"test_min_int16",
                        "test_min_int32",
                        "test_min_int64",
                        #"test_min_int8",
                        #"test_min_one_input",
                        "test_min_two_inputs",
                        #"test_min_uint16",
                        "test_min_uint32",
                        "test_min_uint64",
                        #"test_min_uint8",
                        "test_max_example",
                        ##"test_max_float16",
                        "test_max_float32",
                        "test_max_float64",
                        #"test_max_int16",
                        "test_max_int32",
                        "test_max_int64",
                        #"test_max_int8",
                        "test_max_one_input",
                        "test_max_two_inputs",
                        #"test_max_uint16",
                        "test_max_uint32",
                        "test_max_uint64",
                        #"test_max_uint8",
                        "test_basic_conv_without_padding",
                        "test_conv_with_autopad_same",
                        "test_relu",
                        "test_leakyrelu",
                        "test_leakyrelu_default",
                        "test_leakyrelu_example"]

            onnx_output = outputs(prefix*dirname)[1] # TODO: some tests have more than 1 output
            model, inputs = load_model_and_inputs(prefix*dirname)
            julia_output = play!(model, inputs...)
            @test onnx_output≈julia_output
            # save and eval test
            ort_test(model, inputs...) #from ort.jl
        end
    end
end
