import ONNX: take

@testset "Ops" begin
    @testset "Take" begin
        data = [1.0 2.3 4.5;
                1.2 3.4 5.7]
        idxs = [1 2 1;
                2 3 3]
        @test take(data, idxs) == cat(
            [1.0 2.3; 1.2 3.4],
            [2.3 4.5; 3.4 5.7],
            [1.0 4.5; 1.2 5.7];
            dims=3
        )

        idxs = [1 1 2;
                1 2 2]
        out = take(data, idxs; dim=1)
        @test out[:, 1, :] == [1.0 2.3 4.5; 1.0 2.3 4.5]
        @test out[:, 2, :] == [1.0 2.3 4.5; 1.2 3.4 5.7]
        @test out[:, 3, :] == [1.2 3.4 5.7; 1.2 3.4 5.7]

        idxs = [1, 2, 1]
        @test take(data, idxs) == [1.0 2.3 1.0; 1.2 3.4 1.2]

        idxs = [2]
        @test take(data, idxs) == [1.2]
    end
end