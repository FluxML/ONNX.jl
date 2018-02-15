# TODO: we need kwarg support for many of these

# Generic

ops[:Concat] = function (params, xs...)
  vcall(:cat, params[:axis], xs...)
end

ops[:Gemm] = function (params, A, B, C)
  @assert !haskey(params, :alpha) && !haskey(params, :beta)
  A = get(params, :transA, 0) == 1 ? vcall(:transpose, A) : A
  B = get(params, :transB, 0) == 1 ? vcall(:transpose, B) : B
  vcall(broadcast, :+, vcall(*, B, A), C)
end

# Image

function pads(ps)
  padbegin = ps[1:end÷2]
  padend   = ps[end÷2+1:end]
  padbegin == padend || error("Only symmetric padding currently supported, got $padbegin and $padend")
  return (padbegin...,)
end

ops[:Conv] = function (params, x, w, b)
  length(params[:kernel_shape]) == 2 || error("Only Conv2D currently supported")
  vcall(vcall(:Conv2D, w, b, pads(params[:pads]), (params[:strides]...,)), x)
end

ops[:MaxPool] = function (params, x)
  vcall(:maxpool, x, (params[:kernel_shape]...,), pads(params[:pads]), (params[:strides]...,))
end

ops[:GlobalAveragePool] = function (params, x)
  vcall(:mean, x, (1,2))
end

# Regularise

ops[:Dropout] = function (params, x)
  vcall(vcall(:Dropout, params[:ratio]), x)
end

# Activation

ops[:Relu] = function (params, x)
  vcall(broadcast, :relu, x)
end

ops[:Softmax] = function (params, x)
  vcall(:softmax, x)
end
