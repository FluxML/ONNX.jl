@testset "Save and Load" begin
    @testset "Basic ops" begin
        args = (rand(3, 4), rand(3, 4))
        ort_test(ONNX.add, args...)
    end
end