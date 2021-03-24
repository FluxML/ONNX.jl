@testset "Read and write" begin
    import ONNX

    function serdeser(p::T) where T
        iob = PipeBuffer();
        ONNX.writeproto(iob, p)
        return ONNX.readproto(iob, T())
    end

    @testset "TensorProto" begin
        import ONNX: TensorProto, array

        @testset "Tensor type $T size $s" for T in (Int8, Int32, Int64, Float16, Float32, Float64), s in ((1,),
        (1, 2),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5))
            exp = reshape(collect(T, 1:prod(s)), s...)
            @test TensorProto(exp) |> serdeser |> array == exp
        end
    end

    @testset "ValueInfo" begin
        import ONNX: ValueInfoProto

        @testset "ValueInfo shape $s" for s in ((), (missing,), (1, 2), (3, 4, missing))

            vip = ValueInfoProto("test", s)

            dvip = serdeser(vip)

            @test dvip.name == vip.name

            vsize = size(dvip)
            @test length(vsize) == length(s)
            if !isempty(s)
                @test vsize[findall(!ismissing, s)] == Tuple(skipmissing(s))
            end
        end
    end

    @testset "Attribute" begin
        import ONNX: AttributeProto, TensorProto, attribute, array
    
        @testset "Attribute type $(first(p))" for p in (
        :Int64 => 12,
        :Float32 => 23f0,
        :Float32s => Float32.(1:4),
        :Int64s => [1, 2, 3, 4],
        :String => "relu",
        :Strings => split("abcdefg", "")
        )
            @test AttributeProto(p) |> serdeser |> attribute == p
        end

        @testset "Attribute type Float64" begin
            # Float64 does not exist as attribute type in ONNX so above test will fail with rounging errors
            @test AttributeProto(:ff => 1.23) |> serdeser |> attribute |> last == 1.23f0
        end

        @testset "Attribute type TensorProto" begin
            # TensorProto has undef fields which mess up straigh comparison
            arr = collect(1:4)
            @test AttributeProto(:ff => TensorProto(arr)) |> serdeser |> attribute |> last |> array == arr 
        end

        @testset "Attribute Dict" begin
            attrs = Dict(:int => 12, :str => "aaa", :floats => Float32.(2:5))
            @test pairs(attrs) |> collect .|> AttributeProto .|> serdeser |> Dict == attrs
        end
    end
end