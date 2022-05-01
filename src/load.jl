using Ghost
using Ghost: Tape, Input, Constant, mkcall, Variable, V


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


function load_node!(tape::Tape, ::OpConfig{:ONNX, :GlobalAveragePool}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, global_average_pool, args...)
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


function load_node!(tape::Tape, ::OpConfig{:ONNX, :BatchNormalization},
        args::VarVec, attrs::AttrDict)
    kw = from_onnx_norm(attrs)
    bn = push_call!(tape, batch_norm, args...; kw...)
    if bn._op.val isa Tuple
        # usual in training mode
        # unpack tuples into calls to getfield
        y = push_call!(tape, getfield, bn, 1)
        μnext = push_call!(tape, getfield, bn, 2)
        σ²next = push_call!(tape, getfield, bn, 3)
        return y, μnext, σ²next
    else
        return bn
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Shape}, args::VarVec, attrs::AttrDict)
    # TODO: handle start and end attributes
    return push_call!(tape, size_vector, args[1])
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Constant}, args::VarVec, attrs::AttrDict)
    val_attr = first(keys(attrs))
    val = if val_attr == :value
        array(attrs[val_attr])
    else
        error("Don't know how to load constant value from attribute $val_attr")
    end
    return push!(tape, Constant(val))
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Gather}, args::VarVec, attrs::AttrDict)
    axis = get(attrs, :axis, 0)
    data = tape[args[1]].val
    dim = ndims(data) - axis
    return push_call!(tape, onnx_gather, args...; dim=dim)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Unsqueeze}, args::VarVec, attrs::AttrDict)
    if length(args) == 2
        # ONNX >= v13
        return push_call!(tape, onnx_unsqueeze, args...)
    elseif length(args) == 1
        # ONNX < v13
        axes = attrs[:axes]
        v_axes = push!(tape, Constant(axes))
        return push_call!(tape, onnx_unsqueeze, args[1], v_axes)
    else
        throw(ArgumentError("Cannot load node from Unsqueeze with $(length(args)) arguments"))
    end
end

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
function load(io::IO, args...; backends=[:ONNX], exec::Bool=true)
    onnx_model = readproto(io, ModelProto());
    g = onnx_model.graph;
    tape = Tape(ONNXCtx(backends; exec=exec))
    # create map of initializers
    init_vals = Dict{String, Any}(init.name => array(init)
        for init in g.initializer)
    # load inputs; if input has init value, take it
    # otherwise take the next available argument value
    arg_idx = 1
    used_init_names = Set([])
    for inp in g.input
        val = get(init_vals, inp.name, missing)
        v = V(0)   # will be overwritten
        if val === missing && exec == true
            @assert(
                arg_idx <= length(args),
                "Neither initializer, nor argument is provided for input $(inp.name)"
            )
            val = args[arg_idx]
            arg_idx += 1
            v = push!(tape, Input(val))
        else
            # convert inputs that also have initializers to constants
            # these are usually model parameters, but may
            v = push!(tape, Constant(val))
        end
        tape.c.name2var[inp.name] = v
        push!(used_init_names, inp.name)
    end
    # load the rest of initilizers as constants
    for init in g.initializer
        name = init.name
        if !in(name, used_init_names)
            val = init_vals[name]
            v = push!(tape, Ghost.Constant(val))
            tape.c.name2var[name] = v
        end
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
        vars = [tape.c.name2var[o.name] for o in g.output]
        @assert(all(tape[v] isa Call && tape[v].fn == getfield for v in vars),
            "Don't understand this multi-output result of the graph")
        tape.result = tape[vars[1]].args[1]
    end
    return tape
end

function load(filename::String, args...; backends=[:ONNX], exec::Bool=true)
    return open(filename) do io
        load(io, args...; backends=backends, exec=exec)
    end
end