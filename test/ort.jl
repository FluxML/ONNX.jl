using Test
import ONNXRunTime as OX
import Ghost: play!, Tape, Input, Constant
import ONNX: ONNXCtx, push_call!, from_nnlib, from_onnx, save, load


function ort_run(path, ort_args...)
    model = OX.load_inference(OX.testdatapath(path))
    ort_inputs = Dict([OX.input_names(model)[i] => ort_args[i] for i=1:length(ort_args)])
    return model(ort_inputs)
end


function ort_test(tape::Tape, args...)
    mktemp() do path, _
        r1 = play!(tape, args...)
        save(path, tape)
        r2 = ort_run(path, from_nnlib.(args)...) |> values |> first |> from_onnx
        tape2 = load(path, args...; exec=true)
        r3 = tape2[tape2.result].val
        @test isapprox(r1, r2)
        @test isapprox(r1, r3)
    end
end


function ort_test(fn::Function, args...; kwargs...)
    tape = Tape(ONNXCtx())
    inp = [push!(tape, Input(arg)) for arg in args]
    res = push_call!(tape, fn, inp...; kwargs...)
    tape.result = res
    ort_test(tape, args...)
end