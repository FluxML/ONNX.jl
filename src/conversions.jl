##############################################################################
#                                Tensors                                     #
##############################################################################

"""
    julia2onnx(x)

Convert argument from Julia-friendly to ONNX-friendly format.
The reverse operation is available as [`onnx2julia`](@ref).

See also: [`onnx2julia_conv`](@ref), [`onnx2julia_spatial`](@ref),
[`onnx2julia_conv`](@ref) and similar.
"""
julia2onnx(x::AbstractArray) = permutedims(x, ndims(x):-1:1)

"""
    onnx2julia(x)

The reverse of [`julia2onnx`](@ref).
"""
onnx2julia(x::AbstractArray) = permutedims(x, ndims(x):-1:1)


##############################################################################
#                             Conv Attributes                                #
##############################################################################

"""
    julia2onnx_spatial(x)

Convert spetial attributes such as Conv's stride or dilation
from Julia to ONNX format
"""
julia2onnx_spatial(x) = x
julia2onnx_spatial(x::Tuple) = collect(reverse(x))

onnx2julia_spatial(x) = x
onnx2julia_spatial(x::AbstractVector) = Tuple(reverse(x))


function julia2onnx_pad(pad::Int, N::Int)
    return [pad for i=1:2N]
end

function julia2onnx_pad(pad::NTuple{T, K}, N::Int) where {T, K}
    @assert(K == N || K == 2N, "Padding should be either a tuple of N or 2N elements")
    if K == 2N
        pad = [pad; pad]
    end
    return [pad[N:-1:1]; pad[2N:-1:N+1]]
end

function onnx2julia_pad(pad::Vector{Int})
    # In ONNX, `pads` is always a list of size 2N such as
    # [x1_begin, x2_begin...x1_end, x2_end,...].
    # In NNlib (the default backend for Conv), `pad` expects either Int,
    # or list of Ints of size N or 2N.
    # So ONNX -> Julia is straghtforward except for the reversed order of dimensions.
    # [x1_begin, x2_begin, x1_end, x2_end] => [x2_begin, x1_begin, x1_end, x2_end]
    N = length(pad) ÷ 2
    out = [pad[N:-1:1]; pad[2N:-1:N+1]]
    return out
end


function julia2onnx_conv(attrs::Dict)
    out = Dict{Symbol, Any}()
    if haskey(attrs, :stride)
        out[:strides] = attrs[:stride] |> julia2onnx_spatial
    end
    if haskey(attrs, :dilation)
        out[:dilations] = attrs[:dilation] |> julia2onnx_spatial
    end
    if haskey(attrs, :groups)
        out[:group] = attrs[:groups]
    end
    if haskey(attrs, :pad)
        out[:pads] = attrs[:pad] |> julia2onnx_pad
    end
    return out
end


function onnx2julia_conv(attrs::Dict)
    out = Dict{Symbol, Any}()
    if haskey(attrs, :strides)
        out[:stride] = attrs[:strides] |> onnx2julia_spatial
    end
    if haskey(attrs, :dilations)
        out[:dilation] = attrs[:dilations] |> onnx2julia_spatial
    end
    if haskey(attrs, :group)
        out[:groups] = attrs[:group]
    end
    if haskey(attrs, :pads)
        out[:pad] = attrs[:pads] |> onnx2julia_pad
    end
    return out
end