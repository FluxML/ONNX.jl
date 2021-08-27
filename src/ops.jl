conv(x, w; kw...) = NNlib.conv(x, w; kw...)

function conv(x, w, b; kw...)
    bias_size = (ntuple(_ -> 1, length(kw.stride))..., :, 1)
    b = reshape(b, bias_size)
    return conv(x, w; kw...) .+ b
end


function gemm(A, B, C; tA=0, tB=false, α=1, β=1)
    A = Bool(tA) ? A' : A
    B = Bool(tB) ? B' : B
    return α * A * B .+ β * C
end

function gemm(A, B; tA=0, tB=0, α=1)
    A = Bool(tA) ? A' : A
    B = Bool(tB) ? B' : B
    return α * A * B
end

# Julia-friendly flatten
function flatten(x; dim=ndims(x) - 1)
    sz = size(x)
    keep = dim < ndims(x) ? sz[dim + 1:end] : 1
    return reshape(x, :, keep...)
end

# ONNX-specific flatten
function onnx_flatten(x; axis=1)
    dim = axis >= 0 ? axis + 1 : ndims(x) - axis + 1
    return flatten(x; dim=dim)

end

add(xs...) = broadcast(+, xs...)
mul(xs...) = broadcast(*, xs...)
relu(x) = broadcast(NNlib.relu, x)
maxpool(x, k; pad, stride) = NNlib.maxpool(x, k; pad=pad, stride=stride)

# TODO: implement hese functions
batch_norm(args...) = error("Not implemented")
global_average_pool(args...) = error("Not implemented")