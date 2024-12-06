# Default implementations of ONNX operators

import NNlib
import Statistics: mean
import StaticArrays: SVector

using LinearAlgebra

flipweights(w::AbstractArray{T,N}) where {T,N} = w[(size(w, i):-1:1 for i = 1:(N-2))..., :, :]

conv(x, w; kw...) = NNlib.conv(x, flipweights(w); kw...)

function conv(x, w, b; kw...)
    d = ndims(x) - 2
    bias_size = (ntuple(_ -> 1, d)..., :, 1)
    b = reshape(b, bias_size)
    return conv(x, w; kw...) .+ b
end


function onnx_gemm(A, B, C; tA = 0, tB = 0, α = 1, β = 1)
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

add(xs...) = .+(xs...)
sub(xs...) = .-(xs...)
_sin(x) = sin.(x)
_cos(x) = cos.(x)
mul(xs...) = .*(xs...)
relu(x) = NNlib.relu.(x)
leakyrelu(x;a = 0.01) = NNlib.leakyrelu.(x,a)
elu(x) = NNlib.elu.(x)
tanh(x) = Base.tanh.(x)
maxpool(x; kernel, pad = 0, stride = 1) = NNlib.maxpool(x, kernel; pad = pad, stride = stride)
_min(xs...) = min.(xs...)
_max(xs...) = max.(xs...)

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


size_vector(x) = SVector(size(x))


"""
    take(data, idxs; dim=ndims(data))

Take elements from an array along an axis. For example, for a 4D data
and dim=3, it is roughly equivalent to `data[:, :, idxs, :]`, but allows
multidimensional idxs. See `numpy.take` for a more detailed explanation
of the concept.

In the context of ONNX, `take` is used to implement Gather operation.
We do NOT record this function directly to the tape during loading though,
but instead use a more ONNX-friendly wrapper `onnx_gather()`.

Note: in ONNX, Gather is different from GatherElements, GatherND and
Julia's `NNlib.gather()`.
"""
function take(
        data::AbstractArray{T, N}, idxs::AbstractArray{Int, M};
        dim=ndims(data)) where {T, N, M}
    if length(idxs) == 1 && data isa SVector
        # we use SVector to represent array size, Gather(arr_sz, idx)
        # works as size(arr, idx); but since dimensions are reversed,
        # we need to reverse the index as well
        # see https://github.com/FluxML/ONNX.jl/issues/62 for details
        return data[length(data) .- idxs .+ 1]
    end
    if length(idxs) == 1
        # special case, works as getindex
        return data[idxs]
    end
    # we will take slices of data of this size
    size_before = (size(data)[1:dim-1]...,)
    size_after = (size(data)[dim+1:ndims(data)]...,)
    # and put them into output array at out[:, :, ..., idxs[i, j, ...]]
    out = similar(data, (size_before..., size(idxs)..., size_after...))
    colons_before = [(:) for _=1:dim-1]
    colons_after = [(:) for _=dim+1:ndims(data)]
    # iteration over idxs doesn't depend on data or dimension
    # we iterate over the last index purely due to memory layout
    for i=1:size(idxs, ndims(idxs))
        # R - slice of idxs (not slice of data!)
        R = [[(:) for _=1:ndims(idxs)-1]..., i]
        # ensure I = idxs[R...] is itself an array and not a scalar
        I = [idxs[R...]...,]
        slice = data[colons_before..., I, colons_after...]
        out[colons_before..., R..., colons_after...] = slice
    end
    return out
end

take(data::AbstractArray, idxs::Integer; dim=ndims(data)) =
    take(data, [idxs]; dim=dim)


"""
    onnx_gather(data::AbstractArray, idxs::AbstractArray{Int}; dim=ndims(data))

Implemntation of ONNX's Gather operation with 0-based indices.
For a Julia-friendly version, see `take`.
"""
function onnx_gather(
        data::AbstractArray{T, N}, idxs::AbstractArray{Int, M};
        dim=ndims(data)) where {T, N, M}
    @assert all(idxs .>= 0) "Gather on negative indices is not implemented yet"
    idxs_adjusted = idxs .+ 1
    return take(data, idxs_adjusted; dim=dim)
end


# julia-friendly
function NNlib.unsqueeze(x::AbstractArray, dims)
    new_shape = collect(size(x))
    for d in sort(collect(dims))
        insert!(new_shape, d, 1)
    end
    return reshape(x, new_shape...)
end


# ONNX-friendly, e.g. axes is 0-based, row-major
function onnx_unsqueeze(x::AbstractArray, axes::Vector)
    # ndims(data) + length(axes) => size of the array after unsqueezing
    # .- axes                    => to reverse dimensions
    # .+ 1                       => to convert to 1-based indexing
    # .- 1                       => correction by 1
    dims = ndims(x) + length(axes) .- axes
    return NNlib.unsqueeze(x, dims)
end


function onnx_slice(
        data::AbstractArray,
        starts::VecOrMat{Int}, ends::VecOrMat{Int},
        axes::Vector{Int}=Int[], steps::Vector{Int}=Int[])
    axes = isempty(axes) ? collect(0:ndims(data)-1) : axes
    steps = isempty(steps) ? [1 for i=1:ndims(data)] : steps
    @assert all(starts .>= 0) "Negative indices are not supported yet"
    @assert all(ends .>= 0) "Negative indices are not supported yet"
    # construct ranges, adjusting starts to 1-based indexing
    ranges = [s+1 : st : e for (s, st, e) in zip(starts, steps, ends)]
    # reversed, 1-based dimensions
    dims = ndims(data) .- axes
    # dimension => range mapping
    d2r = Dict(zip(dims, ranges))
    I = [get(d2r, i, (:)) for i=1:ndims(data)]
    return data[I...]
end

# ONNX version of concat, axis is zero-based
function onnx_concat(arrays...; axis)
    @assert length(arrays) >= 1
    dims = axis >= 0 ? ndims(first(arrays)) - axis : -axis 
    return cat(arrays...; dims)
end

function onnx_split(input::AbstractArray; axis)
    dims = axis >= 0 ? ndims(input) - axis  : -axis 
    return onnx_split(input, ones(Int, size(input, dims)); axis)
end

# ONNX version of split, axis is zero-based
function onnx_split(input::AbstractArray, split::Vector{Int}; axis)
    dims = axis >= 0 ? ndims(input) - axis : -axis
    @assert sum(split) == size(input, dims)
    before = Tuple((:) for _ in 1:dims-1)
    after = Tuple((:) for _ in dims+1:ndims(input))
    cumsplit = cumsum(split)
    return Tuple(
        getindex(input, before..., c-s+1:c, after...)
        for (s, c) in zip(split, cumsplit)
    )
end