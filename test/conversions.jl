import ONNX: from_nnlib, from_onnx


@testset "Conversions" begin

    for n=1:4
        x = rand([5 for i=1:n]...)
        xo = from_nnlib(x)
        # test conversion forward and back
        @test from_onnx(xo) == x

        # test elements in x[k1, k2, ..., kn] are equal to xo[kn, ..., k2, k1]
        idx = ntuple(i -> rand(1:size(x, i)), n)
        @test x[idx...] == xo[reverse(idx)...]
    end
end