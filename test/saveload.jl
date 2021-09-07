@testset "Save and Load" begin
    @testset "Basic ops" begin
        args = (rand(3, 4), rand(3, 4))
        ort_test(ONNX.add, args...)
    end

    @testset "Conv" begin
        # 2D
        args = (rand(Float32, 32, 32, 3, 1), rand(Float32, 3, 3, 3, 6))
        ort_test(ONNX.conv, args...)
        ort_test(ONNX.conv, args...; pad=1, stride=(1, 1), dilation=(1, 1), groups=1)
        # ort_test(ONNX.conv, args...; pad=(1, 2), stride=(1, 1), dilation=(1, 1), groups=1)



    end
end