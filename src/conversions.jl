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
    julia2onnx_spatial(x, d)

Convert spatial attributes such as Conv's stride or dilation
from Julia to ONNX format.

`x` is a Julia values of that attribute
`d` is the number of spatial dimensions
"""
julia2onnx_spatial(x::Int, d::Int) = [x for i=1:d]
julia2onnx_spatial(x::Tuple, d::Int) = collect(reverse(x))

onnx2julia_spatial(x) = x
onnx2julia_spatial(x::AbstractVector) = Tuple(reverse(x))


function julia2onnx_pad(pad::Int, N::Int)
    return Tuple([pad for i=1:2N])
end


function julia2onnx_pad(pad::NTuple{N, T}, d::Int) where {N, T}
    @assert(N == d || N == 2d,
        "Padding should be a tuple of either `N` or 2N elements where N is the number " *
        "of spatial dimensions, but got `pad = $pad` and N = $d")
    pad = collect(pad)
    if N != 2d
        # e.g. [1, 2] -> [1, 1, 2, 2]
        pad = repeat(pad, inner=2)
    end
    # notation: (1, 2, 3) - dimensions; b - beginning, e - end of dimension
    # our goal (for 3D case): [b1, e1, b2, e2, b3, e3] -> [b3, b2, b1, e3, e2, e1]
    d = length(pad) ÷ 2
    return [pad[2d-1:-2:1]; pad[2d:-2:2]]
end

function onnx2julia_pad(pad::Vector{Int})
    # notation: (1, 2, 3) - dimensions; b - beginning, e - end of dimension
    # our goal (for 3D case): [b3, b2, b1, e3, e2, e1] -> [b1, e1, b2, e2, b3, e3]
    d = length(pad) ÷ 2
    # step 1: [b3, b2, b1, e3, e2, e1] -> [b1, b2, b3, e1, e2, e3]
    out = [pad[d:-1:1]; pad[2d:-1:d+1]]
    # step 2: [b1, b2, b3, e1, e2, e3] -> [(b1, e1), (b2, e2), (b3, e3)]
    out = [(out[i], out[d + i]) for i=1:d]
    # step 3: [(b1, e1), (b2, e2), (b3, e3)] -> [b1, e1, b2, e2, b3, e3]
    out = reduce((x, y) -> [x...; y...], out)
    return Tuple(out)
end


function julia2onnx_conv(attrs::Dict, d::Int)
    out = Dict{Symbol, Any}()
    if haskey(attrs, :stride)
        out[:strides] = julia2onnx_spatial(attrs[:stride], d)
    end
    if haskey(attrs, :dilation)
        out[:dilations] = julia2onnx_spatial(attrs[:dilation], d)
    end
    if haskey(attrs, :groups)
        out[:group] = attrs[:groups]
    end
    if haskey(attrs, :pad)
        out[:pads] = julia2onnx_pad(attrs[:pad], d)
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