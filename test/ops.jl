import ONNX: onnx_gather

@testset "Ops" begin
    @testset "ONNX Gather" begin
        data = [1.0 2.3 4.5;
                1.2 3.4 5.7]
        idxs = [1 2 1;
                2 3 3]
        @test onnx_gather(data, idxs) == cat(
            [1.0 2.3; 1.2 3.4],
            [2.3 4.5; 3.4 5.7],
            [1.0 4.5; 1.2 5.7];
            dims=3
        )

        idxs = [1 1 2;
                1 2 2]
        @test onnx_gather(data, idxs; dim=1) == cat(
            [1.0 1.0; 2.3 2.3; 4.5 4.5],
            [1.0 1.2; 2.3 3.4; 4.5 5.7],
            [1.2 1.2; 3.4 3.4; 5.7 5.7];
            dims=3
        )

        idxs = [1, 2, 1]
        @test onnx_gather(data, idxs) == [1.0 2.3 1.0; 1.2 3.4 1.2]
    end
end