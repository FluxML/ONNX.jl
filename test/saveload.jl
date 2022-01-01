@testset "Save and Load" begin
    @testset "Basic ops" begin
        args = (rand(3, 4), rand(3, 4))
        ort_test(ONNX.add, args...)
        ort_test(ONNX.mul, args...)
    end

    @testset "Gemm" begin
        A, B, C = (rand(3, 4), rand(3, 4), rand(3, 3))
        ort_test(ONNX.onnx_gemm, A, B')
        ort_test(ONNX.onnx_gemm, A', B)
        ort_test(ONNX.onnx_gemm, A', B, C)
        ort_test(ONNX.onnx_gemm, A, B, C; tA=1)
        ort_test(ONNX.onnx_gemm, A, B; tB=1)
        ort_test(ONNX.onnx_gemm, A', B; α=2.0)
        ort_test(ONNX.onnx_gemm, A', B, C; α=2.0, β=0.5)
        # make sure Gemm with just 2 matrices and no keyword arguments
        # is recorded as just *
        before, after = ort_test(*, A', B)
        @test before[V(3)].fn == after[V(3)].fn
        @test before[V(3)].fn == *
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

        ort_test(ONNX.global_average_pool, x)
    end

    @testset "Activations" begin
        x = rand(3, 4)
        ort_test(ONNX.relu, x)
    end

    @testset "Normalization" begin
        x = rand(7, 7, 3, 5); γ = rand(3); β = rand(3); μ = rand(3); σ² = rand(3)
        ort_test(ONNX.batch_norm, x, γ, β, μ, σ²)
        ort_test(ONNX.batch_norm, x, γ, β, μ, σ²; ϵ=1e-4)


        x = rand(7, 7, 3, 5); γ = rand(3); β = rand(3); μ = rand(3); σ² = rand(3)
        args = (x, γ, β, μ, σ²); model_args = args
        tape = Tape(ONNXCtx())
        inp = [push!(tape, Input(a)) for a in args]
        bn = push_call!(tape, ONNX.batch_norm, inp...; training=true)
        y = push_call!(tape, getfield, bn, 1)
        mu = push_call!(tape, getfield, bn, 2)
        s2 = push_call!(tape, getfield, bn, 3)
        tape.result = bn

        ort_test(tape, args...; atol=1e-4)
    end

end