using Base
# TODO: we need kwarg support for many of these

# Generic
get_tuple(x) = (x...,)
get_tuple() = nothing
convert_type(x) = Base.convert(Array{Float32, 1}, x)

ops[:Concat] = function (params, xs...)
  vcall(:cat, params[:axis] + 2, xs...)
end

ops[:Gemm] = function (params, A, B, C)
  @assert haskey(params, :alpha) && haskey(params, :beta)
  layer = DataFlow.isconstant(B)
  A = get(params, :transA, 0) == 1 ? vcall(transpose, A) : A
  B = get(params, :transB, 0) == 1 ? vcall(transpose, B) : B
  layer ?
    vcall(vcall(:Dense, B, C), A) :
    vcall(:broadcast, :+, vcall(*, B, A), C)
end

# Image

function pads(ps)
  padbegin = ps[1:end÷2]
  padend   = ps[end÷2+1:end]
  if (padbegin != padend)
    println("WARNING: RESHAPING PADS DUE TO ASYMMETRIC PADDING")
    ele = Int64(sum(ps) / 4)
    padbegin = (ele, ele)
    return padbegin
  end
  return (padbegin...)
end

ops[:Conv] = function (params, x, w, b...)
  length(params[:kernel_shape]) == 2 || error("Only Conv2D currently supported")
  if !haskey(params, Symbol("pads"))
    params[:pads] = (0,0)
  end
  if !haskey(params, Symbol("strides"))
    params[:strides] = (1,1)
  end
  if (haskey(params, Symbol("auto_pad")))
    if (String(params[:auto_pad]) == "SAME_UPPER" || String(params[:auto_pad] == "SAME_LOWER"))
      params[:pads] =  Base.convert(Array{Int64,1}, (params[:kernel_shape] .- 1)./2) # Only for strides = [1,1]
    end                                                                           # To Do: Add support for other stride values.
  end
  if isempty(b)
    return vcall(vcall(:Conv, w, convert_type([0]), :relu, Symbol("stride=$((params[:strides]...,))"), Symbol("pad=$((params[:pads]...))")), x)
  end
  vcall(vcall(:Conv, w, b[1], Symbol("stride=$((params[:strides]...,))"),Symbol("pad=$(pads(params[:pads]))")), x)
end

ops[:MaxPool] = function (params, x)
  length(params[:kernel_shape]) == 2 || error("Only maxpool2d currently supported")
  strides = params[:strides] == params[:kernel_shape] ? [] : [params[:strides]]
  vcall(:maxpool, x, (params[:kernel_shape]...,), Symbol("pad=$(pads(params[:pads]))"),Symbol("stride=$((params[:strides]...))"))
end

ops[:GlobalAveragePool] = function (params, x)
  vcall(:mean, x, (1,2))
end

ops[:AveragePool] = function (params, x)
  length(params[:kernel_shape]) == 2 || error("Only maxpool2d currently supported")
  strides = params[:strides] == params[:kernel_shape] ? [] : [params[:strides]]
  vcall(:meanpool, x ,(params[:kernel_shape]...), Symbol("pad=$(pads(params[:pads]))"),Symbol("stride=$((params[:strides]...))"))
end

ops[:BatchNormalization] = function (params, x, scale, b, mean, var)
  vcall(vcall(:BatchNorm, Symbol("ϵ=$(params[:epsilon])"),Symbol("momentum=$(params[:momentum])")), x)
end

# Regularise

ops[:Dropout] = function (params, x)
  vcall(vcall(:Dropout, params[:ratio]), x)
end

# Activation

iscallp(f, v) = DataFlow.iscall(v) && f(v[1])
islayer(v, name) = iscallp(l -> iscallp(x -> x == constant(name), l), v)

ops[:Identity] = function(params, x)
  vcall(:identity, x)
end

ops[:Relu] = function (params, x)
  if islayer(x, :Conv) || islayer(x, :Dense)
    layer = x[1]
    layer = vcall(layer[1], layer[2:3]..., :relu, layer[end], layer[4])
    vcall(layer, x[2])
  else
    vcall(broadcast, :relu, x)
  end
end

ops[:LeakyRelu] = function(params, x)
  if !haskey(params, :alpha)
    params[:alpha] = 0.01
  end
  vcall(:leakyrelu, x, params[:alpha])
end

ops[:Sigmoid] = function (params, x)
  vcall(:sigmoid, x)
end

ops[:Softmax] = function (params, x)
  vcall(:softmax, x)
end

ops[:Floor] = function (params, x)
  vcall(:broadcast, vcall(:floor, x))
end

ops[:Exp] = function(params, x)
  vcall(:exp, x)
end

ops[:Log] = function(params, x)
  vcall(:log, x)
end

ops[:Neg] = function(params, x)
  vcall(:*, x, -1)
end

ops[:Constant] = function (params)
  constant(Symbol("weights[\"$(params.name)\"]"))
end

ops[:Reshape] = function(params, tensor)
  vcall(:reshape, tensor, (params[:shape]...))
end

ops[:LRN] = function(params, x)
  vcall(:identity, x)             # Needed: Flux support for LRN
end

#To-Do : add broadcast here (Urgent)
#         Add axis condition here
ops[:Add] = function(params, A, B)
  if haskey(params, :broadcast) && params[:broadcast] == 1
    vcall( :Add,params[:axis], A, B)                  # To-DO : Define Add function  
  else
    # Broadcast not defined: Perform normal addition.
    vcall(:+, A, vcall(:permutedims, B, vcall(:reverse, vcall(:range, 1, vcall(:ndims, B)))))
  end
end

ops[:Mul] = function (params, A, B)
  if (params[:broadcast] == 1)
    vcall( :Mul, params[:axis], A, B)
  else
    vcall(:.*, A, vcall(:permutedims, B ,[2,1]))    # In case of no broadcast, Perform normal Mul operation.
  end
end

ops[:MatMul] = function(params, A, B)
  vcall(:*, A, B)
end

ops[:size] = function(params, A)
  vcall(:prod, vcall(:size, A))
end

ops[:Sqrt] = function(params, A)
  vcall(:broadcast, vcall(:sqrt, A))
end

# Preprocessing

ops[:ImageScaler] = function(params, A)
  if !haskey(params, :scale)
    params[:scale] = 1
  end
  vcall(:.*, A, params[:scale])
end