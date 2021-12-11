@testset "Save and Load" begin
    @testset "Basic ops" begin
        args = (rand(3, 4), rand(3, 4))
        ort_test(ONNX.add, args...)
        ort_test(ONNX.mul, args...)
    end

    @testset "Conv" begin
        # 2D, keywords
        args = (rand(Float32, 32, 32, 3, 1), rand(Float32, 3, 3, 3, 6))
        ort_test(ONNX.conv, args...)
        ort_test(ONNX.conv, args...; pad=1, stride=(1, 1), dilation=(1, 1), groups=1)
        ort_test(ONNX.conv, args...; pad=1, stride=(1, 2), dilation=(2, 1), groups=1)
        ort_test(ONNX.conv, args...; stride=1, dilation=1)
        ort_test(ONNX.conv, args...; pad=(1, 2))
        ort_test(ONNX.conv, args...; pad=(1, 2, 3, 4))

        # 2D, with bias
        ort_test(ONNX.conv, args..., rand(Float32, 6))
        ort_test(ONNX.conv, args..., rand(Float32, 6); pad=(1, 1))

        # 2D, non-square kernel
        args = (rand(Float32, 32, 32, 3, 1), rand(Float32, 5, 3, 3, 6))
        ort_test(ONNX.conv, args...)

        # 1D
        args = (rand(Float32, 32, 3, 1), rand(Float32, 3, 3, 6))
        ort_test(ONNX.conv, args...)
        ort_test(ONNX.conv, args...; pad=(1, 2))

        # 3D
        args = (rand(Float32, 32, 32, 32, 3, 1), rand(Float32, 3, 3, 3, 3, 6))
        ort_test(ONNX.conv, args...)
        ort_test(ONNX.conv, args...; pad=(1, 2, 3))
    end

    @testset "Pooling" begin
        x = rand(Float32, 32, 32, 3, 1)
        k = (2, 2)
        ort_test(ONNX.maxpool, x; kernel=k)
        ort_test(ONNX.maxpool, x; kernel=k, stride=(3, 3))
        ort_test(ONNX.maxpool, x; kernel=k, stride=(3, 3), pad=1)
    end

    @testset "Activations" begin
        x = rand(3, 4)
        ort_test(ONNX.relu, x)
    end

end