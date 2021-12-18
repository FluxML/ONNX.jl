# Default implementations of ONNX operators

import NNlib
import Flux
import Statistics: mean


flipweights(w::AbstractArray{T,N}) where {T,N} = w[(size(w, i):-1:1 for i = 1:(N-2))..., :, :]

conv(x, w; kw...) = NNlib.conv(x, flipweights(w); kw...)

function conv(x, w, b; kw...)
    d = ndims(x) - 2
    bias_size = (ntuple(_ -> 1, d)..., :, 1)
    b = reshape(b, bias_size)
    return conv(x, w; kw...) .+ b
end


function onnx_gemm(A, B, C; tA = 0, tB = false, α = 1, β = 1)
    A = Bool(tA) ? A' : A
    B = Bool(tB) ? B' : B
    # note: order of arguments reversed due to row-major layout
    return α * B * A .+ β * C
end

function onnx_gemm(A, B; tA = 0, tB = 0, α = 1)
    A = Bool(tA) ? A' : A
    B = Bool(tB) ? B' : B
    # note: order of arguments reversed due to row-major layout
    return α * B * A
end

# Julia-friendly flatten
function flatten(x; dim = ndims(x) - 1)
    sz = size(x)
    keep = dim < ndims(x) ? sz[dim+1:end] : 1
    return reshape(x, :, keep...)
end

# ONNX-specific flatten
function onnx_flatten(x; axis = 1)
    dim = axis >= 0 ? ndims(x) - axis + 1 : axis + 1
    return flatten(x; dim = dim)

end

add(xs...) = +(xs...)
mul(xs...) = .*(xs...)
relu(x) = NNlib.relu.(x)
maxpool(x; kernel, pad = 0, stride = 1) = NNlib.maxpool(x, kernel; pad = pad, stride = stride)


# common functional implementation for batch and instance normalization based on
# https://github.com/FluxML/Flux.jl/blob/06970a5fbbb1cb485c5d2cba597a78fb453fc713/src/layers/normalise.jl#L166-L197
function normalization(x::AbstractArray{T,N}, γ, β, μ, σ², reduce_dims, affine_shape;
        ϵ=1e-5, mtm=0.9, training=false) where {T, N}
    # init variables in the function scope instead of the if's scope
    μnext = μ
    σ²next = σ²
    if !training  # testmode
        stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
        μ = reshape(μ, stats_shape)
        σ² = reshape(σ², stats_shape)
    else  # trainmode or testmode without tracked stats
        μold= μ
        σ²old = σ²
        μ = mean(x; dims = reduce_dims)
        σ² = mean((x .- μ) .^ 2; dims = reduce_dims)
        m = prod(size(x, i) for i in reduce_dims)  # needed for computing corrected var
        μnew = vec(N ∈ reduce_dims ? μ : mean(μ, dims = N))
        σ²new = vec(N ∈ reduce_dims ? σ² : mean(σ², dims = N))
        μnext = (1 - mtm) .* μold .+ mtm .* μnew
        σ²next = (1 - mtm) .* σ²old .+ mtm .* (m / (m - one(eltype(σ²old)))) .* σ²new
    end
    out = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    if !isnothing(γ) && !isnothing(β)
        γ = reshape(γ, affine_shape)
        β = reshape(β, affine_shape)
        out = γ .* out .+ β
    end
    if training
        return out, μnext, σ²next
    else
        return out
    end
end


function batch_norm(x::AbstractArray{T,N}, γ, β, μ, σ²;
        ϵ=1e-5, mtm=0.9, training=false) where {T,N}
    reduce_dims = [1:N-2; N]
    affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    return normalization(x, γ, β, μ, σ², reduce_dims, affine_shape;
        ϵ=ϵ, mtm=mtm, training=training)
end


function instance_norm(x::AbstractArray{T,N}, γ, β, μ, σ²;
        ϵ=1e-5, mtm=0.9, training=false) where {T,N}
    reduce_dims = 1:N-2
    affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    return normalization(x, γ, β, μ, σ², reduce_dims, affine_shape;
        ϵ=ϵ, mtm=mtm, training=training)
end





# function _norm_layer_forward(l, x::AbstractArray{T,N}; reduce_dims, affine_shape) where {T,N}
#     if !_isactive(l) && l.track_stats # testmode with tracked stats
#         stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
#         μ = reshape(l.μ, stats_shape)
#         σ² = reshape(l.σ², stats_shape)
#     else  # trainmode or testmode without tracked stats
#         μ = mean(x; dims = reduce_dims)
#         σ² = mean((x .- μ) .^ 2; dims = reduce_dims)
#         if l.track_stats
#             ## update moving mean/std
#             Zygote.ignore() do
#                 mtm = l.momentum
#                 m = prod(size(x, i) for i in reduce_dims)  # needed for computing corrected var
#                 μnew = vec(N ∈ reduce_dims ? μ : mean(μ, dims = N))
#                 σ²new = vec(N ∈ reduce_dims ? σ² : mean(σ², dims = N))
#                 l.μ = (1 - mtm) .* l.μ .+ mtm .* μnew
#                 l.σ² = (1 - mtm) .* l.σ² .+ mtm .* (m / (m - one(eltype(l.σ²)))) .* σ²new
#             end
#         end
#     end
#     if hasaffine(l)
#         γ = reshape(l.γ, affine_shape)
#         β = reshape(l.β, affine_shape)
#         return l.λ.(γ .* (x .- μ) ./ sqrt.(σ² .+ l.ϵ) .+ β)
#     else
#         return l.λ.((x .- μ) ./ sqrt.(σ² .+ l.ϵ))
#     end
# end


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

# function batch_norm(x, γ, β, μ, σ², ϵ, momentum, training_mode)
#     bn = Flux.BatchNorm(
#         identity, β, γ, μ, σ², ϵ, momentum, true, true, nothing, length(γ))
#     y = bn(x)
#     if training_mode
#         return y, bn.μ, bn.σ²  # TODO: are bn.μ and bn.σ² actually updated?
#     else
#         return y
#     end
# end


function global_average_pool(x)
    return Flux.GlobalMeanPool()(x)
end