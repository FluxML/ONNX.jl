
"""
    julia2onnx(x)

Convert argument from Julia-friendly to ONNX-friendly format.
For example:

 * tensors are reshaped from column-major to row-major
 * shape tuples are rearranged accordingly
 * attributes in dicts are renamed, their values are converted too

 The reverse operation is available as [onnx2julia](@ref).
"""
julia2onnx(x::AbstractArray) = reshape(x, reverse(size(x)))


"""
    onnx2julia(x)

The reverse of [julia2onnx](@ref).
"""
onnx2julia(x::AbstractArray) = reshape(x, reverse(size(x)))