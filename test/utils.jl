import ONNX: rename_keys


@testset "Utils" begin
    attrs = Dict(:a => 0, :b => 1)
    @test rename_keys(attrs, Dict(:a => :c)) == Dict(:c => 0, :b => 1)
    @test rename_keys(attrs, Dict(:f => :g)) == attrs
    @test rename_keys(attrs, Dict()) == attrs
end