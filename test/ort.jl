using Test
using PyCall
import Ghost: play!, Tape, Input
import ONNX: ONNXCtx, push_call!, julia2onnx, onnx2julia, save, load


function ort_run(path, args...)
    onnxruntime = pyimport("onnxruntime")
    ort_session = onnxruntime.InferenceSession(path)
    ort_inputs = PyDict(Dict([ort_session.get_inputs()[i].name => args[i] for i=1:length(args)]))
    return ort_session.run(nothing, ort_inputs)
end


function ort_test(tape::Tape, args...)
    mktemp() do path, _
        r1 = play!(tape, args...)
        save(path, tape)
        r2 = ort_run(path, julia2onnx.(args)...)[1] |> onnx2julia
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