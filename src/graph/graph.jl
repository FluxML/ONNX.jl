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
weights(g) = Dict(x.name => x.name for x in g.initializer)

function inputs(g::Proto.GraphProto)
  ws = weights(g)
  i = 0
  Dict(x.name => haskey(ws, x.name) ?
        :(weights[$(x.name)]) :
        inputnode(i += 1)
       for x in g.input), i
end

function graph(g::Proto.GraphProto)
  vs, n = inputs(g)
  for node in g.node
    vs[node.output[1]] = ops[Symbol(node.op_type)](attributes(node.attribute), map(n -> vs[n], node.input)...)
  end
  return vertex(DataFlow.Lambda(n, vs[g.output[1].name])) |> DataFlow.λopen |> DataFlow.λclose
end

code(g::Proto.GraphProto) = graph(g) |> syntax |> MacroTools.prettify

# iscall(x) = isexpr(x, :call) || (isexpr(x, :.) && isexpr(x.args[2], :tuple))

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

# function liftbegin(ex)
#   ls = []
#   for x in ex.args
#     x = MacroTools.prewalk(x) do x
#       isexpr(x, :block) || return x
#       x.args = filter(x -> isexpr(x, :(=)) && (push!(ls, x); false), x.args)
#     end
#     push!(ls, x)
#   end
#   :($(ls...);)
# end
