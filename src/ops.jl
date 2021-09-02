# Default implementations of ONNX operations

import NNlib
import Flux


conv(x, w; kw...) = NNlib.conv(x, w; kw...)

function conv(x, w, b; kw...)
    bias_size = (ntuple(_ -> 1, length(kw.stride))..., :, 1)
    b = reshape(b, bias_size)
    return conv(x, w; kw...) .+ b
end


function onnx_gemm(A, B, C; tA=0, tB=false, α=1, β=1)
    A = Bool(tA) ? A' : A
    B = Bool(tB) ? B' : B
    # note: order of arguments reversed due to row-major layout
    return α * B * A .+ β * C
end

function onnx_gemm(A, B; tA=0, tB=0, α=1)
    A = Bool(tA) ? A' : A
    B = Bool(tB) ? B' : B
    # note: order of arguments reversed due to row-major layout
    return α * B * A
end

# Julia-friendly flatten
function flatten(x; dim=ndims(x) - 1)
    sz = size(x)
    keep = dim < ndims(x) ? sz[dim + 1:end] : 1
    return reshape(x, :, keep...)
end

# ONNX-specific flatten
function onnx_flatten(x; axis=1)
    dim = axis >= 0 ? ndims(x) - axis + 1 : axis + 1
    return flatten(x; dim=dim)

end

add(xs...) = broadcast(+, xs...)
mul(xs...) = broadcast(*, xs...)
relu(x) = broadcast(NNlib.relu, x)
maxpool(x, k; pad, stride) = NNlib.maxpool(x, k; pad=pad, stride=stride)


# mutable struct BatchNorm{F,V,N,W}
#     λ::F  # activation function
#     β::V  # bias
#     γ::V  # scale
#     μ::W     # moving mean
#     σ²::W    # moving var
#     ϵ::N
#     momentum::N
#     affine::Bool
#     track_stats::Bool
#     active::Union{Bool, Nothing}
#     chs::Int # number of channels
#   end

function batch_norm(x, γ, β, μ, σ², ϵ, momentum, training_mode)
    bn = Flux.BatchNorm(
        identity, β, γ, μ, σ², ϵ, momentum, true, true, nothing, length(γ))
    y = bn(x)
    if training_mode
        return y, bn.μ, bn.σ²  # TODO: are bn.μ and bn.σ² actually updated?
    else
        return y
    end
end


function global_average_pool(x)
    return Flux.GlobalMeanPool()(x)
end