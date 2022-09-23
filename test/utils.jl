import ONNX: rename_keys, unpacked_vars
import Umlaut: Tape, Input, mkcall


@testset "Utils" begin
    # rename_keys
    attrs = Dict(:a => 0, :b => 1)
    @test rename_keys(attrs, Dict(:a => :c)) == Dict(:c => 0, :b => 1)
    @test rename_keys(attrs, Dict(:f => :g)) == attrs
    @test rename_keys(attrs, Dict()) == attrs

    # unpacked_vars
    make_tuple(x) = (x, x + 1)

    tape = Tape()
    x = push!(tape, Input(1.0))
    out = push!(tape, mkcall(make_tuple, x))
    y1 = push!(tape, mkcall(getfield, out, 1))
    y2 = push!(tape, mkcall(getfield, out, 2))
    @test unpacked_vars(tape[out]) == [y1, y2]

    tape = Tape()
    x = push!(tape, Input(1.0))
    out = push!(tape, mkcall(make_tuple, x))
    y1 = push!(tape, mkcall(getfield, out, 1))
    @test unpacked_vars(tape[out]) == [y1, nothing]
end
