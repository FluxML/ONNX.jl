using NNlib: conv, relu, softmax, maxpool2d
using Flux: @require, Chain, Dense, RNN, LSTM, GRU,
        Dropout, LayerNorm, BatchNorm,
        SGD, ADAM, Momentum, Nesterov, AMSGrad,
        param, params, mapleaves, cpu, gpu

struct Conv{N,F,A,V}
  weight::A
  bias::V
  σ::F
  pad::NTuple{N,Int}
  stride::NTuple{N,Int}
end

function (c::Conv)(x)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  σ.(conv(x, c.weight, stride = c.stride, pad = c.pad) .+ b)
end


