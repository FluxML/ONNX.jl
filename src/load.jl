using Umlaut
using Umlaut: Tape, Input, Constant, mkcall, Variable, V


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
const VarVec = Vector{Umlaut.Variable}
const AttrDict = Dict{Symbol, Any}

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Sin}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, _sin, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Cos}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, _cos, args[1])
end

function load_node!(tape::Tape, nd::NodeProto, backend::Symbol)
    args = [tape.c.name2var[name] for name in nd.input]
    attrs = convert(Dict{Symbol, Any}, Dict(nd.attribute))
    conf = OpConfig{backend, Symbol(nd.op_type)}()
    try
        out = load_node!(tape, conf, args, attrs)
        ismissing(out) && return out
        if out isa Tuple
            for i=1:length(nd.output)
                tape.c.name2var[nd.output[i]] = out[i]
            end
        else
            tape.c.name2var[nd.output[1]] = out
        end
    catch
        @error "Error while loading node $nd"
        rethrow()
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

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Sub}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, sub, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Mul}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, mul, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Max}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, _max, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Min}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, _min, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Relu}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, relu, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :LeakyRelu}, args::VarVec, attrs::AttrDict)
    haskey
    return push_call!(tape, leakyrelu, args[1]; (;a = get(attrs,:alpha, 0.01))...) #default value 
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Elu}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, elu, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Tanh}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, tanh, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :MatMul}, args::VarVec, attrs::AttrDict)
    A_ndims = ndims(args[1]._op.val)
    B_ndims = ndims(args[2]._op.val)
    if A_ndims == 2 && B_ndims == 2
        return push_call!(tape, *, args[2], args[1])
    elseif A_ndims in (2, 3) && B_ndims in (2, 3)
        return push_call!(tape, NNlib.batched_mul, args[2], args[1])
    else
        error("MatMul with arrays of $A_ndims and $B_ndims is not implemented yet")
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Sigmoid}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, Broadcast.broadcast, NNlib.sigmoid, args...)
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


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Slice}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_slice, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Split}, args::VarVec, attrs::AttrDict)
    axis = get(attrs, :axis, 0)
    split = if haskey(attrs, :split) # Version 1, 2, 11
        attrs[:split]
    elseif length(args) == 2
        args[2]
    else
        # the results cannot be split in multiple outputs on the tape
        # if the output size is not known during tracing.
        error("Unhandled case where split is not provided")
    end
    out = push_call!(tape, onnx_split, first(args), split; axis)
    return Tuple(
        push_call!(tape, getfield, out, i)
        for i in eachindex(split.op.val)
    )
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Concat}, args::VarVec, attrs::AttrDict)
    axis = get(attrs, :axis, 1)
    return push_call!(tape, onnx_concat, args...; axis)
end

###############################################################################
#                                    API                                      #
###############################################################################


"""
    load(io::IO, model_args...; backends=[:ONNX], exec::Bool=true)
    load(filename::String, model_args...; backends=[:ONNX], exec::Bool=true)

Load an ONNX model as a Umlaut.Tape. The way a particular ONNX node is deserialized is
controlled by methods of [load_node!](@ref) dispatched by backend and node's op_type.

`backends` parameter can be used to customize the loading process.

`exec` parameter instructs the loader to execute every added operation just after
the addition, making the debugging easier. Default is `true`.

See also: [`save!`](@ref)
"""
function load(io::IO, args...; backends=[:ONNX], exec::Bool=true)
    onnx_model = decode(ProtoDecoder(io), ModelProto);
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
            v = push!(tape, Umlaut.Constant(val))
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
        tape.result = Umlaut.bound(tape, V(length(tape)))
    else
        vars = [tape.c.name2var[o.name] for o in g.output]
        is_unpacked_tuple = all(
            tape[v] isa Call && tape[v].fn == getfield && tape[v].args[1] === tuple
            for v in vars
        )
        if is_unpacked_tuple
            # tuple output: we expect tape to contain these outputs as vars  destructured
            # from a multi-ouput op using a sequence of `getfield()` calls
            tape.result = tape[vars[1]].args[1]
        else
            # independent vars in the ouput - create a new tuple var
            tape.result = push!(tape, mkcall(tuple, vars...))
        end
    end
    return tape
end

function load(filename::String, args...; backends=[:ONNX], exec::Bool=true)
    return open(filename) do io
        load(io, args...; backends=backends, exec=exec)
    end
end