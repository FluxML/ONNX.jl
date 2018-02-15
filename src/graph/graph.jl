using DataFlow: Call, constant, inputnode, syntax

const ops = Dict{Symbol,Any}()
include("ops.jl")

function attribute(x::Proto.AttributeProto)
  field = [:f, :i, :s, :t, :g, :floats, :ints, :strings, :tensors, :graphs][x._type]
  Symbol(x.name) => getfield(x, field)
end

attributes(as) = Dict(attribute(a) for a in as)

vcall(a...) = vertex(Call(), constant.(a)...)

# Placeholder for array values
weights(g) = Dict(x.name => get_array(x) for x in g.initializer)

function inputs(g::Proto.GraphProto)
  ws = weights(g)
  i = 0
  Dict(x.name => haskey(ws, x.name) ?
        constant(:(weights[$(x.name)])) :
        inputnode(i += 1)
       for x in g.input), i
end

function _graph(g)
  vs, n = inputs(g)
  for node in g.node
    vs[node.output[1]] = ops[Symbol(node.op_type)](attributes(node.attribute), map(n -> vs[n], node.input)...)
  end
  return vs[g.output[1].name], n
end

# Graph Cleanups

ischainable(v) = DataFlow.iscall(v) && all(x -> DataFlow.isconstant(x), v[3:end])
chaindepth(v) = ischainable(v) ? chaindepth(v[2]) + 1 : 0

function _tochain(v, ch)
  ischainable(v) || return v
  if length(v[:]) ≤ 2
    push!(ch, v[1])
  else
    push!(ch, vertex(DataFlow.Lambda(1, vcall(v[1], inputnode(1), v[3:end]...))))
  end
  return _tochain(v[2], ch)
end

function tochain(v)
  ch = []
  v = _tochain(v, ch)
  vcall(vcall(:Chain, reverse(ch)...), v)
end

function chainify(v)
  MacroTools.prewalk(v) do v
    chaindepth(v) > 3 ? tochain(v) : v
  end
end

# Interface

function graph(g::Proto.GraphProto)
  v, n = _graph(g)
  v = chainify(v)
  return vertex(DataFlow.Lambda(n, v)) |> DataFlow.λopen |> DataFlow.λclose
end

code(g::Proto.GraphProto) = graph(g) |> syntax |> MacroTools.prettify

# function breakcalls(ex)
#   MacroTools.prewalk(ex) do ex
#     iscall(ex) || return ex
#     @capture(ex, f_(args__) | f_.(args__))
#     count(x -> iscall(x), args) ≥ 2 || return ex
#     vars = []
#     args = map(args) do x
#       iscall(x) || return x
#       var = gensym()
#       push!(vars, :($var = $x))
#       return var
#     end
#     ex = isexpr(ex, :call) ? :($f($(args...))) : :($f.($(args...)))
#     :($(vars...); $ex)
#   end
# end
