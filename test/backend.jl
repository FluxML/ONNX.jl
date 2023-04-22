@testset "Backend" begin
    import ONNX.ProtoBuf: encode, decode, ProtoEncoder, ProtoDecoder
    import ONNX: load, array
    import Umlaut: Tape, play!

    "filename.pb to julia array"
    function pb_to_array(filename::String)
        pb = decode(ProtoDecoder(open(filename)), TensorProto)
        return array(pb)
    end

        """
        eval_model(dir)

    run onnx backend model using the data in directory.
    directory structure is:
    dir (test_<name>)
    |--model.onnx
    |--test_data_set_0
       |-- input_0.bp
       |-- ...
       |-- input_N.bp
       |-- output_0.bp
    Loads model.onnx and the inputs using ONNX.jl, run it.
    """
    function eval_model(dir::String)
        ## load inputs
        inputs = readdir(dir*"/test_data_set_0",join=true)|>
            f->filter(contains("input"),f).|>
            pb_to_array
        ## load the model
        model = load(dir*"/model.onnx",inputs...)
        ## run it
        return play!(model,inputs...)
    end

    @testset "Nodes" begin
        prefix = "backend/data/node/"
        for dir in ["test_add"]
            onnx_output = pb_to_array(
                prefix*dir*"/test_data_set_0/output_0.pb")
            julia_output = eval_model(prefix*dir)
            @test onnx_output==julia_output
        end
    end
end


