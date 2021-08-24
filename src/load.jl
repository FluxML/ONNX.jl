using Ghost
using Ghost: Tape, Input, mkcall, Variable, V
using NNlib


struct ONNXCtx
    name2var::Dict{String, Variable}
    backends::Vector{Symbol}
    exec::Bool
end

ONNXCtx(backends; exec=true) = ONNXCtx(Dict(), backends, exec)

# TODO: implement rebind_context!()

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


# function load_node!(tape::Tape, nd::NodeProto, ::Val{BE}) where BE
#     load_node!(tape, nd, Val(BE), Val(Symbol(nd.op_type)))
# end


function load_node!(tape::Tape, nd::NodeProto, ::Val{:NNlib}, ::Val{:Conv})
    attrs = Dict(nd.attribute)
    _,_,p,s,d = akpsd(attrs)
    kw = (stride = s, pad = p, dilation = d, groups = get(attrs, "group", 1))
    args = [tape.c.name2var[name] for name in nd.input]
    # record conv
    res = push_call!(tape, NNlib.conv, args[1], args[2]; kw...)
    if length(args) == 3
        # record bias reshaping
        bias_size = (ntuple(_ -> 1, length(s))..., :, 1)
        b = push_call!(tape, reshape, args[3], bias_size)
        # record bias addition
        res = push_call!(tape, broadcast, +, res, b)
    end
    # update name mapping
    tape.c.name2var[nd.output[1]] = res
end


function load_node!(tape::Tape, nd::NodeProto, ::Val{:Base}, ::Val{:Gemm})
    args = [tape.c.name2var[name] for name in nd.input]
    A, B = args[1:2]
    attrs = Dict(nd.attribute)
    if get(attrs, :transA, 0) == 1
        A = push_call!(tape, transpose, A)
    end
    if get(attrs, :transB, 0) == 1
        B = push_call!(tape, transpose, B)
    end
    C = push_call!(tape, *, A, B)
    α, β = get(attrs, :alpha, 1.0), get(attrs, :beta, 1.0)
    if α != 1.0
        C = push_call!(tape, *, α, C)
    end
    if length(args) == 3
        bias = args[3]
        # TODO: do we need reshape here?
        C = push_call!(tape, broadcast, +, C, bias)
        if β != 1.0
            C = push_call!(tape, *, β, C)
        end
    end
    tape.c.name2var[nd.output[1]] = C
end


function load_node!(tape::Tape, nd::NodeProto, ::Val{:Base}, ::Val{:Add})
    args = [tape.c.name2var[name] for name in nd.input]
    r = push_call!(tape, broadcast, +, args[1], args[2])
    tape.c.name2var[nd.output[1]] = r
end


function load_node!(tape::Tape, nd::NodeProto, ::Val{:Base}, ::Val{:Mul})
    args = [tape.c.name2var[name] for name in nd.input]
    r = push_call!(tape, broadcast, *, args[1], args[2])
    tape.c.name2var[nd.output[1]] = r
end


function load_node!(tape::Tape, nd::NodeProto, ::Val{:NNlib}, ::Val{:Relu})
    args = [tape.c.name2var[name] for name in nd.input]
    r = push_call!(tape, broadcast, NNlib.relu, args[1])
    tape.c.name2var[nd.output[1]] = r
end


function load_node!(tape::Tape, nd::NodeProto, ::Val{:NNlib}, ::Val{:MaxPool})
    args = [tape.c.name2var[name] for name in nd.input]
    attrs = Dict(nd.attribute)
    _,k,p,s,d = akpsd(attrs)
    r = push_call!(tape, NNlib.maxpool, args[1], k; pad=p, stride=s)
    tape.c.name2var[nd.output[1]] = r
end


# function load_node!(tape::Tape, nd::NodeProto, ::Val{BE}, ::Val{OP}) where {BE, OP}
#     @warn "add_node!() is not implemented for op_type = $OP, adding dummy operation instead"
#     args = [tape.c.name2var[name] for name in nd.input]
#     r = push!(tape, mkcall(identity, args[1]))
#     tape.c.name2var[nd.output[1]] = r
# end


###############################################################################
#                                    API                                      #
###############################################################################


function load(filename::AbstractString, model_args...; backends=[:Base, :NNlib], exec::Bool=true)
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
    for nd in g.node
        success = false
        for backend in tape.c.backends
            # TODO: test methods using @which or iterating over methods(load_node!)
            try
                load_node!(tape, nd, Val(backend), Val(Symbol(nd.op_type)))
                @warn "Loaded $(nd.op_type) using backend $(backend)"
            catch e
                e isa MethodError && continue
                rethrow()
            end
            success = true
        end
        success || error("Couldn't load node for $(nd.op_type), " *
                        "tried the following backends: $(tape.c.backends)")
    end
    return tape
end