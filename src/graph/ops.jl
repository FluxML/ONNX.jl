# Generic

ops[:Concat] = function (params, xs...)
  vcall(:cat, params[:axis], xs...)
end

# Image

ops[:Conv] = function (params, x, w, b)
  vcall(vcall(:Conv, w, b, (params[:pads]...,), (params[:strides]...,)), x)
end

ops[:MaxPool] = function (params, x)
  vcall(:maxpool2d, x, (params[:kernel_shape]...,), (params[:pads]...,), (params[:strides]...,))
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
