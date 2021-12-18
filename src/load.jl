using Ghost
using Ghost: Tape, Input, mkcall, Variable, V


struct ONNXCtx
    name2var::Dict{String, Variable}
    backends::Vector{Symbol}
    exec::Bool
end

ONNXCtx(backends; exec=true) = ONNXCtx(Dict(), backends, exec)
ONNXCtx(;exec=true) = ONNXCtx(Dict(), [:ONNX], exec)

# TODO: implement rebind_context!()

"""
    getindex(tape::Tape{ONNXCtx}, onnx_name::String)

Get operation on the tape using the name in ONNX graph
"""
Base.getindex(tape::Tape{ONNXCtx}, onnx_name::String) =
    tape[tape.c.name2var[onnx_name]]

###############################################################################
#                               Operations                                    #
###############################################################################

"""
    push_call!(tape::Tape{ONNXCtx}, fn, args...; kwargs)

Shortcut for `push!(tape, mkcall(fn, args..))` also handling
keyword arguments and respecting `ONNXCtx.exec` setting.
"""
function push_call!(tape::Tape{ONNXCtx}, fn, args...; kwargs...)
    kwargs = NamedTuple(kwargs)
    if !isempty(kwargs)
        args = (kwargs, fn, args...)
        fn = Core.kwfunc(fn)
    end
    op = tape.c.exec ? mkcall(fn, args...) : mkcall(fn, args...; val=nothing)
    return push!(tape, op)
end


# A few constants to keep function signatures concise
struct OpConfig{BE, Op} end
const VarVec = Vector{Ghost.Variable}
const AttrDict = Dict{Symbol, Any}


function load_node!(tape::Tape, nd::NodeProto, backend::Symbol)
    args = [tape.c.name2var[name] for name in nd.input]
    attrs = convert(Dict{Symbol, Any}, Dict(nd.attribute))
    conf = OpConfig{backend, Symbol(nd.op_type)}()
    out = load_node!(tape, conf, args, attrs)
    ismissing(out) && return out
    if out isa Tuple
        for i=1:length(nd.output)
            tape.c.name2var[nd.output[i]] = out[i]
        end
    else
        tape.c.name2var[nd.output[1]] = out
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Gemm}, args::VarVec, attrs::AttrDict)
    if (length(args) == 2 && get(attrs, :alpha, 1) == 1 &&
        get(attrs, :transA, 0) == 0 && get(attrs, :transB, 0) == 0)
        # simplified version: just matrix multiplication
        # note: arguments are swapped to account for row-major arrays
        return push_call!(tape, *, args[2], args[1])
    else
        # complete GEMM version
        kw = rename_keys(attrs, Dict(
            :transA => :tA,
            :transB => :tB,
            :alpha => :α,
            :beta => :β
        ))
        return push_call!(tape, onnx_gemm, args...; kw...)
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Conv}, args::VarVec, attrs::AttrDict)
    kw = from_onnx_conv(attrs) |> NamedTuple
    return push_call!(tape, conv, args...; kw...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :MaxPool}, args::VarVec, attrs::AttrDict)
    kw = from_onnx_conv(attrs; pooling=true) |> NamedTuple
    return push_call!(tape, maxpool, args[1]; kw...)
end



# function load_node!(tape::Tape, ::OpConfig{:ONNX, :Flatten}, args::VarVec, attrs::AttrDict)
#     return push_call!(tape, onnx_flatten, args...; attrs...)
# end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Add}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, add, args...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Mul}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, mul, args...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Relu}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, relu, args[1])
end


# function load_node!(tape::Tape, ::OpConfig{:ONNX, :BatchNormalization}, args::VarVec, attrs::AttrDict)
#     ϵ = get(attrs, :epsilon, 1f-5)
#     momentum = get(attrs, :momentum, 9f-1)
#     training_mode = Bool(get(attrs, :training_mode, 0))
#     res = push_call!(tape, batch_norm, args..., ϵ, momentum, training_mode)
#     if training_mode
#         y = push_call!(tape, getfield, 1)
#         μ_new = push_call!(tape, getfield, 2)
#         σ²_new = push_call!(tape, getfield, 3)
#         return y, μ_new, σ²_new
#     else
#         return res
#     end
# end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :BatchNormalization},
        args::VarVec, attrs::AttrDict)
    attrs = copy(attrs)
    if haskey(attrs, :is_test)
        attrs[:training_mode] = 1 - attrs[:is_test]
        delete!(attrs, :is_test)
    end
    if haskey(attrs, :spatial)
        # Reading the doc below, I assume `spatial` attribute has been removed in opset > 14
        # and later versions always use `spatial=1`
        # https://github.com/onnx/onnx/blob/master/docs/Changelog.md#BatchNormalization-14
        @assert attrs[:spatial] == 1 "BatchNormalization with spatial != 1 is not supported"
        delete!(attrs, :spatial)
    end
    kw = rename_keys(attrs, Dict(
        :epsilon => :ϵ,
        :momentum => :mtm,
        :training_mode => :training
    )) |> Dict{Symbol, Any}
    if haskey(kw, :training)
        kw[:training] = Bool(kw[:training])
    end
    bn = push_call!(tape, batch_norm, args...; kw...)
    if bn._op.val isa Tuple
        # usual in training model
        # unpack tuples into calls to getfield
        y = push_call!(tape, getfield, bn, 1)
        μnext = push_call!(tape, getfield, bn, 2)
        σ²next = push_call!(tape, getfield, bn, 3)
        return y, μnext, σ²next
    else
        return bn
    end
end



# function load_node!(tape::Tape, ::OpConfig{:ONNX, :GlobalAveragePool}, args::VarVec, attrs::AttrDict)
#     return push_call!(tape, global_average_pool, args...)
# end


###############################################################################
#                                    API                                      #
###############################################################################


"""
    load(io::IO, model_args...; backends=[:ONNX], exec::Bool=true)
    load(filename::String, model_args...; backends=[:ONNX], exec::Bool=true)

Load an ONNX model as a Ghost.Tape. The way a particular ONNX node is deserialized is
controlled by methods of [load_node!](@ref) dispatched by backend and node's op_type.

`backends` parameter can be used to customize the loading process.

`exec` parameter instructs the loader to execute every added operation just after
the addition, making the debugging easier. Default is `true`.

See also: [`save!`](@ref)
"""
function load(io::IO, model_args...; backends=[:ONNX], exec::Bool=true)
    onnx_model = readproto(io, ModelProto());
    g = onnx_model.graph;
    tape = Tape(ONNXCtx(backends; exec=exec))
    # create map of initializers
    init_vals = Dict{String, Any}()
    for init in g.initializer
        # TODO: consider non-array inputs
        init_vals[init.name] = array(init)
    end
    # load inputs
    arg_idx = 1
    for inp in g.input
        val = get(init_vals, inp.name, missing)
        if val === missing && exec == true
            val = model_args[arg_idx]
            arg_idx += 1
        end
        v = push!(tape, Input(val))
        tape.c.name2var[inp.name] = v
    end
    # load nodes
    for nd in g.node
        success = false
        for backend in tape.c.backends
          if !ismissing(load_node!(tape, nd, backend))
            success = true
            @debug "Loaded $(nd.op_type) using backend $(backend)"
            break
          end
        end
        success || error("Couldn't load node for $(nd.op_type), " *
                         "tried the following backends: $(tape.c.backends)")
    end
    if length(g.output) == 1
        tape.result = Ghost.bound(tape, V(length(tape)))
    else
        # tuple output: we expect tape to contain these outputs as vars  destructured
        # from a multi-ouput op using a sequence of `getfield()` calls
        vars = [tape.c.name2var[name] for name in nd.output]
        @assert(all(tape[v] isa Call && tape[v].fn == getfield for v in vars),
            "Don't understand this multi-output result of the graph")
        tape.result = tape[vars[1]].args[1]
    end
    return tape
end

function load(filename::String, model_args...; backends=[:ONNX], exec::Bool=true)
    return open(filename) do io
        load(io, model_args...; backends=backends, exec=exec)
    end
end