import ONNX: julia2onnx, onnx2julia


@testset "Conversions" begin

    for n=1:4
        x = rand([5 for i=1:n]...)
        xo = julia2onnx(x)
        # test conversion forward and back
        @test onnx2julia(xo) == x

        # test elements in x[k1, k2, ..., kn] are equal to xo[kn, ..., k2, k1]
        idx = ntuple(i -> rand(1:size(x, i)), n)
        @test x[idx...] == xo[reverse(idx)...]
    end
end