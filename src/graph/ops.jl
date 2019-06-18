# This file contains the implementation of various operators.
# Tests for them is at test/runtests.jl.

using Base
using Statistics
# TODO: we need kwarg support for many of these

# Generic
get_tuple(x) = (x...,)
get_tuple() = nothing
convert_type(x) = Base.convert(Array{Float32, 1}, x)

ops[:Concat] = function (params, ip...)
  #s = vcall(:ndims, ip1, ip2, ip3, ip4)

  return vcall(:cat, ip..., Symbol("dims =1"))
end

ops[:Gemm] = function (params, A, B, C)
  if !haskey(params, :transA)
    params[:transA] = 0
  end
  if !haskey(params, :transB)
    params[:transB] = 0
  end
  if !haskey(params, :alpha)
    params[:alpha] = 1
  end
  if !haskey(params, :beta)
    params[:beta] = 1
  end
  if !haskey(params, :broadcast)
    params[:broadcast] = 0
  end
  if (params[:transA] != 1)
    A =  vcall(:permutedims, A, vcall(:reverse, vcall(:range, 1, vcall(:ndims, A))))
  end
  if (params[:transB] != 1)
    B = vcall(:permutedims, B, vcall(:reverse, vcall(:range, 1, vcall(:ndims, B))))
  end
  ip1 = vcall(:*, params[:alpha], A, B)
  ip2 = vcall(:*, params[:beta], C)
  if params[:broadcast] == 0
    ip1 = vcall(:permutedims, ip1, vcall(:reverse, vcall(:range, 1, vcall(:ndims, ip1))))
    res = vcall(:broadcast, :+, ip1, ip2)
    return res
  end
  res = vcall(:broadcast, :+, ip1, ip2)
  return vcall(:permutedims, res, vcall(:reverse, vcall(:range, 1, vcall(:ndims, res))))
end

# Image

ops[:Conv] = function (params, x, w, b...)
  if !haskey(params, Symbol("pads"))
    params[:pads] = [0,0,0,0]
  end
  if !haskey(params, Symbol("strides"))
    params[:strides] = (1,1)
  end
  if !haskey(params, Symbol("dilations"))
    params[:dilations] = (1,1)
  end
  if (haskey(params, Symbol("auto_pad")))
    if (String(params[:auto_pad]) == "SAME_UPPER" || String(params[:auto_pad] == "SAME_LOWER"))
      temp = Base.convert(Array{Int64,1}, (params[:kernel_shape] .- 1)./2) # Only for strides = [1,1]
      params[:pads] = vcat(temp, temp)                                    # To Do: Add support for other stride values.                                                                           
    elseif String(params[:auto_pad]) == "VALID"
      params[:pads] = [0,0,0,0]
    end
  end
  #if haskey(params, :group)
  #  s = vcall(:Int, vcall(:/, vcall(:size, x, 3), params[:group]))
  #  x = vcall(:reshape, x, vcall(:size, x, 1), vcall(:size, x, 2), s, params[:group], vcall(:size, x, 4))
  #  temp_x = vcall(:getindex, x, :,:,:,1,:)
  #  temp = vcall(vcall(:Conv, Float32[0], :relu, 
  #      Symbol("stride=$((params[:strides]...,))"), Symbol("pad=$(pads(params[:pads]))"),
  #        Symbol("dilation=$((params[:dilations]...,))")), temp_x)
  #  if isempty(b)    
  #    for i=2:params[:group]
  #      temp = vcall(:cat, 3, temp, vcall(vcall(:Conv, Float32[0], :relu, 
  #        Symbol("stride=$((params[:strides]...,))"), Symbol("pad=$(pads(params[:pads]))"),
  #          Symbol("dilation=$((params[:dilations]...,))")), temp))
  #    end
  #    
  #  else
  #    for i=2:params[:group]
  #      temp = vcall(:cat, 3, temp, vcall(vcall(:Conv, b[1], :relu, 
  #        Symbol("stride=$((params[:strides]...,))"), Symbol("pad=$(pads(params[:pads]))"),
  #          Symbol("dilation=$((params[:dilations]...,))")), temp))
  #    end
  #  end
  #  return temp
  #end
  if isempty(b)
    return vcall(vcall(:CrossCor, w, Float32[0], :relu, Symbol("stride=$((params[:strides]...,))"),
     Symbol("pad=$((params[:pads]...,))"), Symbol("dilation=$((params[:dilations]...,))")), x)
                                 # temp change (Until type fix)
  end
  vcall(vcall(:CrossCor, w, b[1], Symbol("stride=$((params[:strides]...,))"), 
    Symbol("pad=$((params[:pads]...,))"),  Symbol("dilation=$((params[:dilations]...,))")), x)
end

ops[:MaxPool] = function (params, x)
  if !(haskey(params, :strides))
    params[:strides] = [1,1]
  end
  if !(haskey(params, :pads))
    params[:pads] = [0,0,0,0]
  end
  strides = params[:strides] == params[:kernel_shape] ? [] : [params[:strides]]
  if length(params[:kernel_shape]) == 1
    push!(params[:kernel_shape], 1)
    n_size = vcall(:Tuple, vcall(:push!, vcall(:collect, vcall(:size, x)), 1))
    new_x = vcall(:reshape, x, n_size)
    return vcall(:dropdims, vcall(:maxpool, new_x, (params[:kernel_shape]...,), Symbol("pad=$(params[:pads])"),
        Symbol("stride=$((params[:strides]...,))")), Symbol("dims=4")) 
  end
  
  vcall(vcall(:MaxPool, (params[:kernel_shape]...,), Symbol("pad=$((params[:pads]...,))"),Symbol("stride=$((params[:strides]...,))")), x)
end

ops[:GlobalAveragePool] = function (params, x)
  vcall(:mean, x, Symbol("dims = (1,2)"))
end

ops[:GlobalMaxPool] = function (params, x)
  vcall(:getindex, vcall(:findmax, x, Symbol("dims=(1,2)")), 1)
end

ops[:AveragePool] = function (params, x)
  length(params[:kernel_shape]) <= 2 || error("Only averagepool2d currently supported")
  if !haskey(params, :strides)
    params[:strides] = [1,1]
  end
  strides = params[:strides] == params[:kernel_shape] ? [] : [params[:strides]]
  if !haskey(params, :pads)
    params[:pads] = [0,0,0,0]
  end
  if length(params[:kernel_shape]) == 1
    push!(params[:kernel_shape], 1)
    n_size = vcall(:Tuple, vcall(:push!, vcall(:collect, vcall(:size, x)), 1))
    new_x = vcall(:reshape, x, n_size)
    return vcall(:dropdims, vcall(:meanpool, new_x, (params[:kernel_shape]...,), Symbol("pad=$((params[:pads]...,))"),
        Symbol("stride=$((params[:strides]...,))")), Symbol("dims=4")) 
  end
  if params[:pads] == [0,0,0,0]
    return vcall(vcall(:MeanPool, (params[:kernel_shape]...,), Symbol("pad=$((params[:pads]...,))"),
                                                    Symbol("stride=$((params[:strides]...,))")), x)
  else
    params[:strides_temp] = [1,1]
    params[:kernel_shape_temp] = [1,1]
    params[:pads_temp] = [0,0,0,0]
    temp = vcall(vcall(:MeanPool, (params[:kernel_shape_temp]...,), Symbol("pad=$((params[:pads]...,))"),
                                                    Symbol("stride=$((params[:strides_temp]...,))")), x)
    return vcall(vcall(:MeanPool, (params[:kernel_shape]...,), Symbol("pad=$((params[:pads_temp]...,))"),
                                                    Symbol("stride=$((params[:strides]...,))")), temp)
  end                                               
end

ops[:BatchNormalization] = function (params, x, scale, b, mean, var)
  if !haskey(params ,Symbol("momentum"))
    params[:momentum] = 0.9
  end
  if !haskey(params, Symbol("epsilon"))
    params[:epsilon] = 1e-5
  end
  t = typeof(params[:momentum])
  q = vcall(:broadcast, :+, params[:epsilon], var)
  p = vcall(:broadcast, sqrt ,q)
  r = vcall(:broadcast, Float32, p)
  return vcall(vcall(:BatchNorm,identity, b, scale, vcall(:broadcast, :Float32, mean), r, t(params[:epsilon]), params[:momentum], false), x)
end

function slice(a, s, e)
  return a[s:e]
end

ops[:LSTM] = function(params, ip...)
  if length(ip) == 3
    len = params[:hidden_size]  
    arg1 = vcall(reshape, ip[2], (4*len,2))
    arg2 = vcall(reshape, ip[3], (4*len,3))
    ip_ = vcall(reshape, ip[1], vcall(slice ,vcall(:size, ip[1]), 1, 2))
    
    a = vcall(LSTM, arg1, arg2, zeros(len*4), zeros(len), zeros(len))
    
    return vcall(a, ip_)
  elseif length(ip) == 4
    len = params[:hidden_size]
    arg1 = vcall(reshape, ip[2], (4*len,3))
    arg2 = vcall(reshape, ip[3], (4*len,4))
    arg3 = ip[4][1:4*len]
    b1 = vcall(:broadcast, Float32, vcall(reinterpret, Float32, vcall(zeros, 2)))
    a = vcall(LSTM, arg1, arg2, arg3, b1, b1)
    
    ip_ = vcall(reshape, ip[1], vcall(slice ,vcall(:size, ip[1]), 1, 2))
    return vcall(a, ip_)
  end
end

# Regularise

ops[:Dropout] = function (params, x)
  return vcall(:identity, x)        # Inference mode: Dropout just bypasses input.
end

# Activation

iscallp(f, v) = DataFlow.iscall(v) && f(v[1])
islayer(v, name) = iscallp(l -> iscallp(x -> x == constant(name), l), v)

ops[:Identity] = function(params, x)
  vcall(:identity, x)
end

ops[:Flatten] = function(params, x)
  if !haskey(params, :axis)
    params[:axis] = 1
  end
  l = vcall(:length, x)
  rev = vcall(:reverse, vcall(:size, x))
  if (params[:axis] == 0)
    return vcall(:reshape, x, l, 1)
  else 
    s = vcall(:prod, vcall(:getindex, rev, 1:params[:axis]))
    return vcall(:reshape, x, vcall(:div, l, s), s)
  end
end

ops[:Relu] = function (params, x)
  vcall(broadcast, :relu, x)
  #end
end

ops[:LeakyRelu] = function(params, x)
  if !haskey(params, :alpha)
    params[:alpha] = 0.01
  end
  vcall(:broadcast, :leakyrelu, x, params[:alpha])
end

ops[:PRelu] = function(params, x, slope)
  ip1 = vcall(:broadcast, :clamp, x, 0, Inf)
  ip2 = vcall(:.*, vcall(:broadcast, :clamp, x, -Inf, 0), slope)
  return vcall(:broadcast, Float32, vcall(:+, ip1, ip2))
end

ops[:ArgMax] = function(params, x)
  return vcall(Flux.argmax, x)
end

ops[:Abs] = function (params, x)
  vcall(:broadcast, abs, x)
end

ops[:Clip] = function (params, x)
  if !haskey(params, :min)
    params[:min] = vcall(:getindex, vcall(:findmin, x), 1)
  end
  if !haskey(params, :max)
    params[:max] = vcall(:getindex, vcall(:findmax, x), 1)
  end
  vcall(:broadcast, clamp, x, params[:min], params[:max])
end

ops[:Equal] = function(params, x, y)
  return vcall(:broadcast, :Int, vcall(:broadcast, :isequal, x, y))
end

ops[:Greater] = function(params, x, y)
  return vcall(:broadcast, :Int, vcall(:broadcast, :isless, y, x))
end

ops[:Sigmoid] = function (params, x)
  vcall(:sigmoid, x)
end

ops[:Softmax] = function (params, x)
  vcall(:softmax, vcall(:vec, x))
end

ops[:Floor] = function (params, x)
  vcall(:broadcast, :floor, x)
end

ops[:Exp] = function(params, x)
  vcall(:broadcast, :exp, x)
end

ops[:Log] = function(params, x)
  vcall(:broadcast, :log, x)
end

ops[:Neg] = function(params, x)
  vcall(:*, -1,  x)
end

ops[:Sum] = function (params, x, y...)
  if isempty(y)
    return vcall(:.+, x, 0)
  end
  vcall(:+, x, y[1])
end

ops[:Cast] = function(params, x)
  if (params[:to] == 1)
    return vcall(:broadcast, :Float32, x)
  elseif params[:to] == 10
    return vcall(:broadcast, :Float16, x)
  elseif params[:to] == 11
    return vcall(:broadcast, :Float64, x)
  end
end

ops[:Constant] = function (params)
  constant(Symbol("weights[\"$(params.name)\"]"))
end

ops[:Ceil] = function (params ,x)
  vcall(:broadcast, :ceil, x)
end

ops[:Unsqueeze] = function(params, x)
  l1 = length(params[:axes])
  l2 = vcall(:+, l1, vcall(:ndims, x))
  temp = x
  for ele in params[:axes]
    temp = vcall(Flux.unsqueeze, temp, vcall(:-, vcall(:+, vcall(:ndims, temp), 1), ele))
  end
  return temp
end

ops[:Reshape] = function(params, tensor1, shape...)
  if haskey(params, :shape)
    return vcall(:reshape, tensor1, vcall(:broadcast, Int64, vcall(:Tuple, params[:shape])))
  end
  vcall(:reshape, tensor1, vcall(:broadcast, Int64, vcall(:Tuple, vcall(:reverse, shape[1]))))
end

ops[:Transpose] = function(params ,tensor)
  temp_tens = vcall(:permutedims, tensor, vcall(:reverse, vcall(:range, 1, vcall(:ndims, tensor))))
  order = vcall(:.+, params[:perm], 1)
  l = vcall(:permutedims, temp_tens, order)
  return vcall(:permutedims, l, vcall(:reverse, vcall(:range, 1, vcall(:ndims, l))))
end

ops[:LRN] = function(params, x)
  if !haskey(params, :bias)
    params[:bias] = 1
  end
  if !haskey(params, :alpha)
    params[:alpha] = 1e-4
  end
  if !haskey(params, :beta)
    params[:beta] = 0.75
  end
  return vcall(vcall(:LRNorm, params[:bias], params[:size], params[:alpha], params[:beta]), x)
                               # currently, just bypassing the output
  #return vcall(:.+, 0, x)
end

#To-Do : add broadcast here (Urgent)
#         Add axis condition here
ops[:Add] = function(params, A, B)
  s1 = vcall(:size, A)
  s2 = vcall(:size, B)
  if (s1==s2)
    return vcall(:Add, params[:axis], A, B)
  else
    return vcall(:.+, A, B)
  end
end

ops[:Sub] = function(params, A , B)
  s1 = vcall(:size, A)
  s2 = vcall(:size, B)
  if (s1==s2)
    return vcall(:-, A, B)
  else
    return vcall(:.-, A, B)
  end
end

ops[:Div] = function(params, A , B)
  if (haskey(params, :broadcast) && params[:broadcast] == 1)
    if !haskey(params, :axis)
      return vcall(:./, A, B)
    end
    return vcall( :Div, params[:axis], A, B)              # To-Do define Div function
  else
    return vcall(:./  , A, B)   # In case of no broadcast, Perform normal div operation.
  end
end

ops[:Mul] = function (params, A, B)
  if (haskey(params, :broadcast) && params[:broadcast] == 1)
    if !haskey(params, :axis)
      return vcall(:.*, A, B)
    end
    return vcall( :Mul, params[:axis], A, B)              # To-Do define Mul function
  else
    return vcall(:.*, A, B)   # In case of no broadcast, Perform normal Mul operation.
  end
end

ops[:Pow] = function (params, A, B)
  if (haskey(params, :broadcast) && params[:broadcast] == 1)
    if !haskey(params, :axis)
      return vcall(:.^, A, B)
    end
    return vcall( :Pow, params[:axis], A, B)              # To-Do define Pow function
  else
    return vcall(:.^, A, B)   # In case of no broadcast, Perform normal Power operation.
  end
end

ops[:MatMul] = function(params, A, B)
  #tempa = vcall(:permutedims, A, vcall(:reverse, vcall(:range, 1, vcall(:ndims, A))))
  #tempb = vcall(:permutedims, B, vcall(:reverse, vcall(:range, 1, vcall(:ndims, B))))
  vcall(:*, B, A)
end

ops[:size] = function(params, A)
  vcall(:prod, vcall(:size, A))
end

ops[:Sqrt] = function(params, A)
  vcall(:broadcast, :sqrt, A)
end

ops[:Reciprocal] = function(params, A)
  vcall(:./ , 1, A)
end

ops[:Xor] = function (params, A, B)
  ip1 = vcall(:broadcast, &, vcall(:Array, vcall(:broadcast, Bool, A)), vcall(:Array, 
              vcall(:broadcast, !, vcall(:broadcast, Bool, B))))
  ip2 = vcall(:broadcast, &, vcall(:Array, vcall(:broadcast, Bool, B)), vcall(:Array, 
              vcall(:broadcast, !, vcall(:broadcast, Bool, A))))
  return  vcall(:broadcast, :Int, vcall(:broadcast, |, ip1, ip2))   
end

ops[:And] = function(params, A, B)
  if (haskey(params, :broadcast) && params[:broadcast] == 1)
    if !haskey(params, :axis)
      return vcall(:.*, vcall(:broadcast, :Bool, A), vcall(:broadcast, :Bool, B))
    end
    return vcall( :And, params[:axis], A, B)              # To-Do define And function
  else
    return vcall(:.*, vcall(:broadcast, :Bool, A), vcall(:broadcast, :Bool, B))   # In case of no broadcast, 
                                                                                    #Perform normal And operation.
  end
end

ops[:Or] = function(params, A, B)
  if (haskey(params, :broadcast) && params[:broadcast] == 1)
    if !haskey(params, :axis)
      return vcall(:.+, vcall(:broadcast, :Bool, A), vcall(:broadcast, :Bool, B))
    end
    return vcall( :Or, params[:axis], A, B)              # To-Do define Or function
  else
    return vcall(:.+, vcall(:broadcast, :Bool, A), vcall(:broadcast, :Bool, B))   # In case of no broadcast, 
                                                                                    #Perform normal Or operation.
  end
end

ops[:Expand] = function(params, A, B)
  shape_new = vcall(:reverse, B)
  return vcall(:repeat , A, Symbol("inner=$(vcall(:reverse, B))"))
end 
# Preprocessing

ops[:ImageScaler] = function(params, A)
  if !haskey(params, :scale)
    params[:scale] = 1
  end
  vcall(:.*, A, params[:scale])
end

#Trigonometric ops

ops[:Cos] = function(params, A)
  vcall(:broadcast, :cos, A)
end

ops[:Sin] = function(params, A)
  vcall(:broadcast, :sin, A)
end

ops[:Tan] = function(params, A)
  vcall(:broadcast, :tan, A)
end

ops[:Acos] = function(params, A)
  vcall(:broadcast, :acos, A)
end

ops[:Asin] = function(params, A)
  vcall(:broadcast, :asin, A)
end

ops[:Atan] = function(params, A)
  vcall(:broadcast, :atan, A)
end
