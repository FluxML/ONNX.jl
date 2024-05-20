##############################################################################
#                                Tensors                                     #
##############################################################################

"""
    from_nnlib(x)

Convert argument from NNlib-friendly to ONNX-friendly format.
The reverse operation is available as [`from_onnx`](@ref).

See also: [`from_onnx_conv`](@ref), [`from_onnx_spatial`](@ref),
[`from_onnx_conv`](@ref) and similar.
"""
from_nnlib(x::AbstractArray) = permutedims(x, ndims(x):-1:1)

"""
    from_onnx(x)

The reverse of [`from_nnlib`](@ref).
"""
from_onnx(x::AbstractArray) = permutedims(x, ndims(x):-1:1)


##############################################################################
#                             Conv Attributes                                #
##############################################################################

"""
    from_nnlib_spatial(x, d)

Convert spatial attributes such as Conv's stride or dilation
from Julia to ONNX format.

`x` is a Julia values of that attribute
`d` is the number of spatial dimensions
"""
from_nnlib_spatial(x::Int, d::Int) = [x for i=1:d]
from_nnlib_spatial(x::Tuple, d::Int) = collect(reverse(x))

from_onnx_spatial(x) = x
from_onnx_spatial(x::AbstractVector) = Tuple(reverse(x))


function from_nnlib_pad(pad::Int, N::Int)
    return Tuple([pad for i=1:2N])
end


function from_nnlib_pad(pad::NTuple{N, T}, d::Int) where {N, T}
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

function from_onnx_pad(pad::Vector{Int})
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


function from_nnlib_conv(attrs::Dict, d::Int)
    out = Dict{Symbol, Any}()
    if haskey(attrs, :stride)
        out[:strides] = from_nnlib_spatial(attrs[:stride], d)
    end
    if haskey(attrs, :dilation)
        out[:dilations] = from_nnlib_spatial(attrs[:dilation], d)
    end
    if haskey(attrs, :kernel)
        out[:kernel_shape] = from_nnlib_spatial(attrs[:kernel], d)
    end
    if haskey(attrs, :groups)
        out[:group] = attrs[:groups]
    end
    if haskey(attrs, :pad)
        out[:pads] = from_nnlib_pad(attrs[:pad], d)
    end
    return out
end


function from_onnx_conv(attrs::Dict; pooling=false)
    out = Dict{Symbol, Any}()
    # manually calculating the padding for :autopad
    if haskey(attrs, :auto_pad)
        if !(attrs[:auto_pad] in ["SAME_LOWER", "SAME_UPPER", "VALID","NOTSET"])
            error("auto_pad $(attrs[:auto_pad]) isn't supported;{SAME_LOWER, SAME_UPPER, VALID, NOTSET}")
        end

        if attrs[:auto_pad]=="SAME_LOWER"||attrs[:auto_pad]=="SAME_UPPER"
            pad = div.(attrs[:kernel_shape].-1,2)
            r = (attrs[:kernel_shape].-1).%2
            if attrs[:auto_pad]=="SAME_LOWER"
                out[:pad] = from_onnx_pad([pad;pad]+[r;zero(r)])
            end
            if attrs[:auto_pad]=="SAME_UPPER"
                out[:pad] = from_onnx_pad([pad;pad]+[zero(r);r])
            end
        end
        #if attrs[:auto_pad]=="VALID" end #pad=0 by default
        #if attrs[:auto_pad]=="NOTSET" end ## TODO: check if :pads is set
    end
    if haskey(attrs, :strides)
        out[:stride] = from_onnx_spatial(attrs[:strides])
    end
    if haskey(attrs, :dilations)
        out[:dilation] = from_onnx_spatial(attrs[:dilations])
    end
    # this attribute is only used for pooling, but not conv
    if haskey(attrs, :kernel_shape) && pooling
        out[:kernel] = from_onnx_spatial(attrs[:kernel_shape])
    end
    if haskey(attrs, :group)
        out[:groups] = attrs[:group]
    end
    if haskey(attrs, :pads)
        out[:pad] = from_onnx_pad(attrs[:pads])
    end
    return out
end


function from_nnlib_norm(attrs::Dict{K,V}) where {K,V}
    out = Dict{Symbol, Any}()
    if haskey(attrs, :training)
        out[:training_mode] = Int(attrs[:training])
    end
    if haskey(attrs, :mtm)
        # ONNX calculates new running mean as `mtm * input_mean + (1 - mtm) * current_mean`
        # NNlib/Flux does the opposite, i.e.  `(1 - mtm) * input_mean + mtm * current_mean`
        # thus we reverse the value here
        out[:momentum] = 1f0 - attrs[:mtm]
    end
    if haskey(attrs, :ϵ)
        out[:epsilon] = attrs[:ϵ]
    end
    return out
end


function from_onnx_norm(attrs::Dict{K,V}) where {K,V}
    out = Dict{Symbol, Any}()
    if haskey(attrs, :is_test)
        out[:training] = Bool(1 - attrs[:is_test])
    end
    if haskey(attrs, :training_mode)
        out[:training] = Bool(attrs[:training_mode])
    end
    if haskey(attrs, :spatial) && attrs[:spatial] != 1
        # deprecated attribute, e.g. see
        # https://github.com/onnx/onnx/blob/master/docs/Changelog.md#BatchNormalization-14
        error("BatchNormalization with spatial != 1 is not supported")
    end
    if haskey(attrs, :momentum)
        # ONNX calculates new running mean as `mtm * input_mean + (1 - mtm) * current_mean`
        # NNlib/Flux does the opposite, i.e.  `(1 - mtm) * input_mean + mtm * current_mean`
        # thus we reverse the value here
        out[:mtm] = 1f0 - attrs[:momentum]
    end
    if haskey(attrs, :epsilon)
        out[:ϵ] = attrs[:epsilon]
    end
    return out
end
