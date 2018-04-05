# TODO: we need kwarg support for many of these

# Generic

ops[:Concat] = function (params, xs...)
  vcall(:cat, params[:axis], xs...)
end

ops[:Gemm] = function (params, A, B, C)
  @assert !haskey(params, :alpha) && !haskey(params, :beta)
  layer = DataFlow.isconstant(B)
  A = get(params, :transA, 0) == 1 ? vcall(transpose, A) : A
  B = get(params, :transB, 0) == 1 ? vcall(transpose, B) : B
  layer ?
    vcall(vcall(:Dense, B, C), A) :
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
  vcall(vcall(:Conv, w, b, pads(params[:pads]), (params[:strides]...,)), x)
end

ops[:MaxPool] = function (params, x)
  length(params[:kernel_shape]) == 2 || error("Only maxpool2d currently supported")
  strides = params[:strides] == params[:kernel_shape] ? [] : [params[:strides]]
  vcall(:maxpool, x, (params[:kernel_shape]...,), pad=pads(params[:pads]), stride=strides)
end

ops[:GlobalAveragePool] = function (params, x)
  vcall(:mean, x, (1,2))
end

# Regularise

ops[:Dropout] = function (params, x)
  vcall(vcall(:Dropout, params[:ratio]), x)
end

# Activation

iscallp(f, v) = DataFlow.iscall(v) && f(v[1])
islayer(v, name) = iscallp(l -> iscallp(x -> x == constant(name), l), v)

ops[:Relu] = function (params, x)
  if islayer(x, :Conv) || islayer(x, :Dense)
    layer = x[1]
    layer = vcall(layer[1], :relu, layer[2:3]...,  layer[end], layer[4])
    vcall(layer, x[2])
  else
    vcall(broadcast, :relu, x)
  end
end

ops[:Softmax] = function (params, x)
  vcall(:softmax, x)
end
