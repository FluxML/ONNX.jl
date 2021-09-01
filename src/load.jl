using Ghost
using Ghost: Tape, Input, mkcall, Variable, V


struct ONNXCtx
    name2var::Dict{String, Variable}
    backends::Vector{Symbol}
    exec::Bool
end

ONNXCtx(backends; exec=true) = ONNXCtx(Dict(), backends, exec)

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


mrev(x) = x
mrev(x::AbstractVector) = reverse(x)
prev(x) = x
prev(x::AbstractVector) = reshape(permutedims(reverse(reshape(x, length(x) ÷ 2,:);dims=1)),:)


# mrev = maybe reverse. prev = rearrange padding, e.g. (1,2,1,2) => (2,2,1,1) or (1,2,3,1,2,3) => (3,3,2,2,1,1)
_akpsd(params) = get(params, :activation, identity), mrev(get(params, :kernel_shape, 1)), prev(get(params, :pads, 0)), mrev(get(params, :strides, 1)), mrev(get(params, :dilations, 1))
akpsd(params) = a2t.(_akpsd(params))
a2t(x) = x
a2t(a::AbstractArray) = Tuple(a)


conv_attr_onnx2tape(attrs) = Dict(
    :stride => mrev(get(attrs, :strides, 1)),
    :pad => prev(get(attrs, :pads, 0)),
    :dilation => mrev(get(attrs, :dilations, 1)),
    :groups => get(attrs, :group, 1),
    # kenrnel_shape => mrev(get(params, :kernel_shape, 1)) -- not used in NNlib.conv
)

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
    if out isa Tuple
        for i=1:length(nd.output)
            tape.c.name2var[nd.output[i]] = out[i]
        end
    else
        tape.c.name2var[nd.output[1]] = out
    end
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Conv}, args::VarVec, attrs::AttrDict)
    _,_,p,s,d = akpsd(attrs)
    kw = (stride = s, pad = p, dilation = d, groups = get(attrs, "group", 1))
    return push_call!(tape, conv, args...; kw...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Gemm}, args::VarVec, attrs::AttrDict)
    kw = Dict(
        :tA => get(attrs, :transA, 0),
        :tB => get(attrs, :transB, 0),
        :α => get(attrs, :alpha, 1),
        :β => get(attrs, :beta, 0)
    )
    return push_call!(tape, onnx_gemm, args...; kw...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Flatten}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_flatten, args...; attrs...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Add}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, add, args...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Mul}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, mul, args...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Relu}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, relu, args[1])
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :MaxPool}, args::VarVec, attrs::AttrDict)
    _,k,p,s,_ = akpsd(attrs)
    return push_call!(tape, maxpool, args[1], k; pad=p, stride=s)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :BatchNormalization}, args::VarVec, attrs::AttrDict)
    ϵ = get(attrs, :epsilon, 1f-5)
    momentum = get(attrs, :momentum, 9f-1)
    training_mode = Bool(get(attrs, :training_mode, 0))
    res = push_call!(tape, batch_norm, args..., ϵ, momentum, training_mode)
    if training_mode
        y = push_call!(tape, getfield, 1)
        μ_new = push_call!(tape, getfield, 2)
        σ²_new = push_call!(tape, getfield, 3)
        return y, μ_new, σ²_new
    else
        return res
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :GlobalAveragePool}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, global_average_pool, args...)
end


###############################################################################
#                                    API                                      #
###############################################################################


function load(
        filename::AbstractString, model_args...;
        backends=[:Base, :NNlib, :ONNX], exec::Bool=true)
    onnx_model = open(filename) do io
        readproto(io, ModelProto())
    end;
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
    op_configs = [meth.sig.parameters[3] for meth in methods(load_node!).ms]
    op_configs = [config for config in op_configs if config <: OpConfig]
    for nd in g.node
        success = false
        for backend in tape.c.backends
            # check if there's an implementation for this op in this backend
            if OpConfig{backend, Symbol(nd.op_type)} in op_configs
                load_node!(tape, nd, backend)
                success = true
                break
                @debug "Loaded $(nd.op_type) using backend $(backend)"
            end
        end
        success || error("Couldn't load node for $(nd.op_type), " *
                        "tried the following backends: $(tape.c.backends)")
    end
    tape.result = Ghost.bound(tape, V(length(tape)))
    return tape
end