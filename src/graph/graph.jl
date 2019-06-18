using DataFlow: Call, constant, inputnode, syntax

const ops = Dict{Symbol,Any}()
include("ops.jl")

# This is to fetch weights when they are stored in
# the constant tensor and not in intializer.
function get_weights(g::Types.Graph)
  temp = Dict{Any, Any}()
  for node in g.node
    if node.op_type == "Constant"
      temp[node.name] = get_array(node.attribute[:value])
    end
  end
  return temp
end

vcall(a...) = vertex(Call(), constant.(a)...)

# Placeholder for array values
weights(f::Types.Model) = weights(f.graph)
  
"""
Checks location of weights and returns appropriate
values. 
Note: Constant weight is deprecated now.
"""
function weights(g::Types.Graph)
  count = 0
  for node in g.node
    if (node.op_type == "Constant")
      count = count + 1
      break
    end  
  end
  if (count > 0)
    return get_weights(g)
  end
  return g.initializer
end

function inputs(g::Types.Graph)
  ws = weights(g)
  i = 0
  Dict(x.name => haskey(ws, x.name) ?
        constant(:(weights[$(x.name)])) :
        inputnode(i += 1)
       for x in g.input), i
end

function _graph(g::Types.Graph)
  vs, n = inputs(g)
  for node in g.node
    if node.op_type == "Constant"
      vs[node.output[1]] = ops[Symbol(node.op_type)](node, map(n -> vs[n], node.input)...)
    else
      vs[node.output[1]] = ops[Symbol(node.op_type)](node.attribute, map(n -> vs[n], node.input)...)
    end
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
function graph(g::Types.Graph)
  v, n = _graph(g)
  v = chainify(v)
  return vertex(DataFlow.Lambda(n, v)) |> DataFlow.λopen |> DataFlow.λclose
end

"""
Write out the Julia code for the model
"""
code(g::Types.Graph) = graph(g) |> syntax |>
  MacroTools.flatten |> MacroTools.gensym_ids

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
