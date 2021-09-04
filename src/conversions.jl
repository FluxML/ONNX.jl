"Maybe reverse"
mrev(x) = x
mrev(x::AbstractVector) = reverse(x)

"""
Convert padding between ONNX convolution (cross-correlation) and
NNlib (proper) convolution
"""
prev(x) = x
prev(x::AbstractVector) = reshape(
    permutedims(
        reverse(
            reshape(
                x,
                length(x) ÷ 2,
                :
            )
            ; dims=1
        )
    ),
    :
)


# mrev = maybe reverse. prev = rearrange padding, e.g. (1,2,1,2) => (2,2,1,1) or (1,2,3,1,2,3) => (3,3,2,2,1,1)
_akpsd(params) = get(params, :activation, identity), mrev(get(params, :kernel_shape, 1)), prev(get(params, :pads, 0)), mrev(get(params, :strides, 1)), mrev(get(params, :dilations, 1))
akpsd(params) = a2t.(_akpsd(params))
a2t(x) = x
a2t(a::AbstractArray) = Tuple(a)


conv_attr_onnx2tape(attrs) = Dict(
    :stride => mrev(get(attrs, :strides, 1)),
    :pad => prev(get(attrs, :pads, 0)),
    :dilation => mrev(get(attrs, :dilations, 1)),
    :groups => get(attrs, :group, 1),
    # kenrnel_shape => mrev(get(params, :kernel_shape, 1)) -- not used in NNlib.conv
)


"""
    julia2onnx(x)

Convert argument from Julia-friendly to ONNX-friendly format.
For example:

 * tensors are reshaped from column-major to row-major
 * shape tuples are rearranged accordingly
 * attributes in dicts are renamed, their values are converted too

 The reverse operation is available as [`onnx2julia`](@ref).
"""
julia2onnx(x::AbstractArray) = permutedims(x, ndims(x):-1:1)



"""
    onnx2julia(x)

The reverse of [`julia2onnx`](@ref).
"""
onnx2julia(x::AbstractArray) = permutedims(x, ndims(x):-1:1)

function onnx2julia(attrs::Dict)
    out = Dict{Symbol, Any}()
    if haskey(attrs, :stride)
        # TODO:
        out[:strides] = mrev(attrs[:stride])
    end

end