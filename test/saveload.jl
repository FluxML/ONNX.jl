import ONNX: graphproto, modelproto, encode
import ONNX: NodeProto, ValueInfoProto, AttributeProto, onnx_name


@testset "Save and Load" begin
    @testset "Multioutput" begin
        args = rand(3, 4), rand(3, 4)
        tape = Tape(ONNXCtx())
        inp = [push!(tape, Input(arg)) for arg in args]
        out1 = push_call!(tape, ONNX.add, inp...)
        out2 = push_call!(tape, ONNX.mul, inp...)
        res = push_call!(tape, tuple, out1, out2)
        tape.result = res
        ort_test(tape, args...)
    end

    @testset "Basic ops" begin
        args = (rand(3, 4), rand(3, 4))
        ort_test(ONNX.add, args...)
        ort_test(ONNX.mul, args...)
    end

    @testset "Sin" begin
        A = rand(3, 4)
        ort_test(ONNX._sin, A)
    end

    @testset "Cos" begin
        A = rand(3, 4)
        ort_test(ONNX._cos, A)
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

    @testset "MatMul" begin
        ort_test(NNlib.batched_mul, rand(3, 4, 5), rand(4, 3))
        ort_test(NNlib.batched_mul, rand(3, 4), rand(4, 3, 5))
        ort_test(NNlib.batched_mul, rand(3, 4, 5), rand(4, 3, 5))

        # 2D*2D case; since it's already covered by Gemm, we have to
        # manually construct the graph
        g = graphproto("generated_model")
        a = Input(rand(3, 4)); a.id = 1
        push!(g.input, ValueInfoProto(a))
        b = Input(rand(4, 5)); b.id = 2
        push!(g.input, ValueInfoProto(b))
        c = mkcall(*, V(a), V(b)); c.id = 3
        nd = NodeProto(
            input=[onnx_name(b), onnx_name(a)],
            output=[onnx_name(c)],
            name=onnx_name(c),
            attribute=AttributeProto[],
            op_type="MatMul"
        )
        push!(g.node, nd)
        push!(g.output, ValueInfoProto(c))
        m = modelproto(g);
        mktemp() do path, io
            encode(ProtoEncoder(io), m)
            seek(io, 0)
            r2_onnx = ort_run(path, from_nnlib(a.val), from_nnlib(b.val))
            r2 = from_onnx(first(values(r2_onnx)))
            @test c.val ≈ r2
            @test c.val ≈ load(path, a.val, b.val)[V(3)].val
        end
    end

    @testset "Conv" begin
        # 2D, keywords
        args = (rand(Float32, 32, 32, 3, 1), rand(Float32, 3, 3, 3, 6))
        ort_test(ONNX.conv, args...)
        ort_test(ONNX.conv, args...; pad=1, stride=(1, 1), dilation=(1, 1), groups=1)
        ort_test(ONNX.conv, args...; pad=1, stride=(1, 2), dilation=(2, 1), groups=1)
        ort_test(ONNX.conv, args...; pad=(3, 3, 3, 3), groups=1, stride=(2, 2), dilation=(1, 1))
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

    @testset "Flatten" begin
        x = rand(Float32, 32, 32, 3, 1)
        ort_test(ONNX.onnx_flatten, x)
    end

    @testset "Activations" begin
        x = rand(3, 4)
        ort_test(ONNX.relu, x)
        # ort_test(ONNX.elu, x) # TODO: Elu is not implemented in ONNXRuntime.jl
        ort_test(ONNX.tanh, x)
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

    @testset "Shape" begin
        # TODO

    end

    @testset "Gather" begin
        data = [1.0 2.3 4.5;
                1.2 3.4 5.7]
        idxs = [1 2 1;
                2 3 3] .- 1
        ort_test(ONNX.onnx_gather, data, idxs)

        idxs = [1 1 2;
                1 2 2] .- 1
        ort_test(ONNX.onnx_gather, data, idxs; dim=1)

        idxs = [1, 2, 1] .- 1
        ort_test(ONNX.onnx_gather, data, idxs)

        data = [3, 4]   # e.g. size of array
        idxs = [2] .- 1
        ort_test(ONNX.onnx_gather, data, idxs)
    end


    @testset "Unsqueeze" begin
        ort_test(ONNX.onnx_unsqueeze, rand(2, 3, 4), [0, 4])
        ort_test(ONNX.onnx_unsqueeze, rand(2, 3, 4), [0, 3])
        ort_test(ONNX.onnx_unsqueeze, [4.0], [0])
    end


    @testset "Slice" begin
        ort_test(ONNX.onnx_slice, rand(5, 10, 20), [0, 0], [3, 10], [0, 1], [1, 1])
        ort_test(ONNX.onnx_slice, rand(5, 10, 20), [0, 0, 0], [3, 10, 5])
        ort_test(ONNX.onnx_slice, rand(5, 10, 20), [3, 0], [0, 10], [0, 1], [1, -1])
    end

    @testset "Concat" begin
        ort_test(ONNX.onnx_concat, [1, 2, 3], [4, 5, 6]; axis=0)
        ort_test(ONNX.onnx_concat, [1, 2, 3], [4, 5, 6]; axis=-1)

        ort_test(ONNX.onnx_concat, [1 2 3; 1 2 3], [4  5; 4 5]; axis=0)
    end

    @testset "Split" begin
        x = rand(3, 20, 10); split = [5, 10, 5];
        args = (x, split)
        tape = Tape(ONNXCtx())
        inp = [push!(tape, Input(a)) for a in args]
        out = push_call!(tape, ONNX.onnx_split, inp...; axis=1)
        push_call!(tape, getfield, out, 1)
        push_call!(tape, getfield, out, 2)
        push_call!(tape, getfield, out, 3)
        tape.result = out

        ort_test(tape, args...)
    end
end