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
    return Tuple([pad for i=1:2N])
end

function julia2onnx_pad(pad::NTuple{N, T}, ND::Int) where {N, T}
    @assert(N == ND || N == 2ND, "Padding should be either a tuple of N or 2N elements")
    if N != 2ND
        pad = [pad...; pad...]
    end
    return Tuple([pad[N:-1:1]; pad[2N:-1:N+1]])
end

function onnx2julia_pad(pad::Vector{Int})
    # In ONNX, `pads` is always a list of size 2N such as
    # [x1_begin, x2_begin...x1_end, x2_end,...].
    # In NNlib (the default backend for Conv), `pad` expects either Int,
    # or list of Ints of size N or 2N.
    # So ONNX -> Julia is straghtforward except for the reversed order of dimensions.
    # [x1_begin, x2_begin, x1_end, x2_end] => [x2_begin, x1_begin, x1_end, x2_end]
    N = length(pad) ÷ 2
    out = Tuple([pad[N:-1:1]; pad[2N:-1:N+1]])
    return out
end


function julia2onnx_conv(attrs::Dict, N::Int)
    out = Dict{Symbol, Any}()
    if haskey(attrs, :stride)
        out[:strides] = julia2onnx_spatial(attrs[:stride])
    end
    if haskey(attrs, :dilation)
        out[:dilations] = julia2onnx_spatial(attrs[:dilation])
    end
    if haskey(attrs, :groups)
        out[:group] = attrs[:groups]
    end
    if haskey(attrs, :pad)
        out[:pads] = julia2onnx_pad(attrs[:pad], N)
    end
    return out
end


function onnx2julia_conv(attrs::Dict)
    out = Dict{Symbol, Any}()
    haskey(attrs, :auto_pad) && error("auto_pad attribute is currently not supported")
    if haskey(attrs, :strides)
        out[:stride] = onnx2julia_spatial(attrs[:strides])
    end
    if haskey(attrs, :dilations)
        out[:dilation] = onnx2julia_spatial(attrs[:dilations])
    end
    if haskey(attrs, :group)
        out[:groups] = attrs[:group]
    end
    if haskey(attrs, :pads)
        out[:pad] = onnx2julia_pad(attrs[:pads])
    end
    return out
end