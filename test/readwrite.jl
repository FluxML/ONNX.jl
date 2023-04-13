@testset "Read and write" begin
    import ONNX
    import ONNX.ProtoBuf: encode, decode, ProtoEncoder, ProtoDecoder
    import Random: MersenneTwister

    function serdeser(p::T) where T
        iob = PipeBuffer();
        encode(ProtoEncoder(iob), p)
        return decode(ProtoDecoder(iob), T)
    end

    @testset "Row-/Columns-major" begin
        import ONNX: TensorProto, array

        # column-major, internally - [1, 2, 3, 4, 5, 6]
        x = Float32[1 4;
                    2 5;
                    3 6]

        p = TensorProto(x)
        # row-major, will be read in e.g. python as
        # [[1 2 3],
        #  [4 5 6]]
        @test p.dims == [2, 3]
        @test p.float_data == Float32[1, 2, 3, 4, 5, 6]

        # column-major again
        x_ = array(p)
        @test x == x_
    end

    @testset "TensorProto" begin
        import ONNX: TensorProto, array
        @testset "Tensor type $T size $s" for T in (Float32,
                                                    UInt8,
                                                    Int8,
                                                    UInt16,
                                                    Int16,
                                                    Int32,
                                                    Int64,
                                                    Bool,
                                                    Float16,
                                                    Float64,
                                                    UInt32,
                                                    UInt64),
            s in ((1,),
                  (1, 2),
                  (1, 2, 3),
                  (1, 2, 3, 4),
                  (1, 2, 3, 4, 5))
            #exp = reshape(collect(T, 1:prod(s)), s...)
            exp = rand(MersenneTwister(0),T,s)
            @test TensorProto(exp) |> serdeser |> array == exp
        end
    end

    @testset "TensorProto(String)" begin
        for exp in (["ONNX"],
                    ["Julia1","Julia2","Julia3"])
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
