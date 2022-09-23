using Test
import ONNXRunTime as OX
import Umlaut: play!, Tape, Input
import ONNX: ONNXCtx, push_call!, from_nnlib, from_onnx, save, load


function ort_run(path, ort_args...)
    model = OX.load_inference(path)
    ort_inputs = Dict([OX.input_names(model)[i] => ort_args[i] for i=1:length(ort_args)])
    return model(ort_inputs)
end


function ort_test(tape::Tape, args...; atol=0)
    mktemp() do path, _
        r1 = play!(tape, args...)
        save(path, tape)
        r2_onnx = ort_run(path, from_nnlib.(args)...)
        r2 = r1 isa Tuple ?  # handle multi-output graphs as well
            map(from_onnx, Tuple(values(r2_onnx))) :
            from_onnx(first(values(r2_onnx)))
        tape2 = load(path, args...; exec=true)
        r3 = tape2[tape2.result].val
        @test r1 isa Tuple ? all(isapprox.(r1, r2; atol=atol)) : isapprox(r1, r2; atol=atol)
        @test r1 isa Tuple ? all(isapprox.(r1, r3; atol=atol)) : isapprox(r1, r3; atol=atol)
        # for more flexibility we return the tape before saving and after loading
        return tape, tape2
    end
end


function ort_test(fn::Function, args...; kwargs...)
    tape = Tape(ONNXCtx())
    inp = [push!(tape, Input(arg)) for arg in args]
    res = push_call!(tape, fn, inp...; kwargs...)
    tape.result = res
    return ort_test(tape, args...)
end