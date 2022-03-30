# Default implementations of ONNX operators

import NNlib
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
        ϵ=1f-5, mtm=0.1f0, training=false) where {T, N}
    # init variables in the function scope instead of the if's scope
    μnext = μ
    σ²next = σ²
    if !training  # testmode
        stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
        μ = reshape(μ, stats_shape)
        σ² = reshape(σ², stats_shape)
    else  # trainmode or testmode without tracked stats
        μold = μ
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
        ϵ=1f-5, mtm=0.1f0, training=false) where {T,N}
    reduce_dims = [1:N-2; N]
    affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    return normalization(x, γ, β, μ, σ², reduce_dims, affine_shape;
        ϵ=ϵ, mtm=mtm, training=training)
end


function instance_norm(x::AbstractArray{T,N}, γ, β, μ, σ²;
        ϵ=1f-5, mtm=0.1f0, training=false) where {T,N}
    reduce_dims = 1:N-2
    affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    return normalization(x, γ, β, μ, σ², reduce_dims, affine_shape;
        ϵ=ϵ, mtm=mtm, training=training)
end


# implementation from
# https://github.com/FluxML/Flux.jl/blob/f66be896d3d2698ce77ce8b7788b4317285bf0b2/src/layers/conv.jl#L605-L614
function global_average_pool(x)
    # Input size
    x_size = size(x)
    # Kernel size
    k = x_size[1:end-2]
    # Pooling dimensions
    pdims = NNlib.PoolDims(x, k)
    return NNlib.meanpool(x, pdims)
end


function onnx_gather(
        data::AbstractArray{T, N}, idxs::AbstractArray{Int, M};
        dim=ndims(data)) where {T, N, M}
    # we will take slices of data of this size
    data_size_except_dim = (size(data)[1:dim-1..., dim+1:ndims(data)...]...,)
    # and put them into output array at out[:, :, ..., idxs[i, j, ...]]
    out = similar(data, (data_size_except_dim..., size(idxs)...))
    # iteration over idxs doesn't depend on data or dimension
    # we iterate over the last index purely due to memory layout
    for i=1:size(idxs, ndims(idxs))
        # R - slice of idxs (not slice of data!)
        R = [[(:) for _=1:ndims(idxs)-1]..., i]
        # ensure I = idxs[R...] is itself an array and not a scalar
        I = [idxs[R...]...,]
        slice = data[[(:) for _=1:dim-1]..., I, [(:) for _=dim+1:ndims(data)]...]
        # move target dimension to the end to confo
        slice = permutedims(slice, [(1:dim-1)..., (dim+1:ndims(data))..., dim])
        out[:, R...] = slice
    end
    return out
end