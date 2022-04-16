import Pkg

modelproto(;kwargs...) = ModelProto(;
    ir_version=7,
    opset_import=[OperatorSetIdProto(version=14)],
    producer_name="ONNX.jl",
    producer_version=string(Pkg.Types.Context().env.project.version), # TODO: Ugh....
    kwargs...)


"""
    graphproto()
Return an [`ONNX.GraphProto`](@ref) with all fields initialized to empty arrays.
"""
graphproto(;kwargs...) = GraphProto(;
    node = NodeProto[],
    initializer = TensorProto[],
    input = ValueInfoProto[],
    output = ValueInfoProto[],
    value_info = ValueInfoProto[],
    kwargs...
)


add!(gp::GraphProto, np::NodeProto) = push!(gp.node, np)

add!(gp::GraphProto, tp::TensorProto) = push!(gp.initializer, tp)


##############################################################################
#                                 Utils                                    #
##############################################################################

# can we make it more robust?
iskwfunc(f) = endswith(string(f), "##kw")

function kwargs2dict(op::Ghost.Call)
    kw = iskwfunc(op.fn) ? op.args[1] : (;)
    return Dict(zip(keys(kw), values(kw)))
end

macro opconfig_kw(backend, fn)
    return quote
        $OpConfig{$backend, <:Union{typeof($fn), typeof(Core.kwfunc($fn))}}
    end
end

function NodeProto(op_type::String, op::Ghost.Call, attrs::Dict=Dict())
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    return NodeProto(
        input=[onnx_name(v) for v in args],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto.(keys(attrs), values(attrs)),
        op_type=op_type
    )
end

ValueInfoProto(op::Ghost.AbstractOp) = ValueInfoProto(
    onnx_name(op),
    # utils in write.jl reverse the shape, so we don't do it here
    # try the following for example:
    #     TypeProto_Tensor((4, 3), Float64).shape.dim[1].dim_value
    # which gives 3 instead of 4
    size(op.val),
    eltype(op.val)
)

##############################################################################
#                                 Methods                                    #
##############################################################################

onnx_name(v::Variable) = "x$(v.id)"
onnx_name(op::Ghost.AbstractOp) = "x$(op.id)"


"""
    save_node!(g::GraphProto, op::Ghost.Call)
    save_node!(g::GraphProto, ::OpConfig{:Backend, Fn}, op::Ghost.Call)

Serialize a single operation from a tape to graph.
"""
function save_node!(g::GraphProto, op::Ghost.Call)
    save_node!(g, OpConfig{:ONNX, typeof(op.fn)}(), op)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(getfield)}, op::Ghost.Call)
    # Do nothing: getfield is only used to destructure multi-ouput nodes
    # and doesn't need to be written to ONNX graph.
    # Using getfield() for anything other then destructuring is thus a mistake.
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(*)}, op::Ghost.Call)
    nd = NodeProto(
        input=[onnx_name(v) for v in reverse(op.args)],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto[],
        op_type="Gemm"
    )
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_gemm), op::Ghost.Call)
    kw_dict = kwargs2dict(op)
    attrs = rename_keys(kw_dict, Dict(
        :tA => :transA,
        :tB => :transB,
        :α => :alpha,
        :β => :beta
    ))
    nd = NodeProto("Gemm", op, attrs)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, conv), op::Ghost.Call)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    w = args[2]._op.val
    # ONNXRuntime gives the following error for Float64:
    # NOT_IMPLEMENTED : Could not find an implementation for the node x3:Conv(11)')
    eltype(w) == Float64 && @warn "Not all ONNX runtimes support input & weights as Float64"
    attrs = from_nnlib_conv(kwargs2dict(op), ndims(w) - 2)
    nd = NodeProto("Conv", op, attrs)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, maxpool), op::Ghost.Call)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    x = args[1]._op.val
    attrs = from_nnlib_conv(kwargs2dict(op), ndims(x) - 2)
    nd = NodeProto("MaxPool", op, attrs)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(global_average_pool)}, op::Ghost.Call)
    nd = NodeProto("GlobalAveragePool", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_flatten), op::Ghost.Call)
    nd = NodeProto("Flatten", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(add)}, op::Ghost.Call)
    nd = NodeProto("Add", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(mul)}, op::Ghost.Call)
    nd = NodeProto("Mul", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(relu)}, op::Ghost.Call)
    nd = NodeProto("Relu", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, batch_norm), op::Ghost.Call)
    kw_dict = kwargs2dict(op)
    attrs = from_nnlib_norm(kw_dict)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    output = if Bool(get(attrs, :training_mode, 0))
        vars = unpacked_vars(op)
        @assert(all([v isa V for v in vars]),
            "Not all output vars of batch_norm are unpacked to the tape")
        [onnx_name(v) for v in vars]
    else
        [onnx_name(op)]
    end
    nd = NodeProto(
        input=[onnx_name(v) for v in args],
        output=output,
        name=onnx_name(op),
        attribute=AttributeProto.(keys(attrs), values(attrs)),
        op_type="BatchNormalization"
    )
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(size)}, op::Ghost.Call)
    nd = NodeProto("Shape", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, <:Any}, op::Ghost.Constant)
    @assert(
        op.val isa AbstractArray,
        "ONNX.jl currently doesn't support saving constants of type $(typeof(op.val))"
    )
    attr_name = :value
    attr_value = from_nnlib(op.val)
    nd = NodeProto(
        input=[],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto.([attr_name], [attr_value]),
        op_type=op_type
    )
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_gather), op::Ghost.Call)
    data = iskwfunc(op.fn) ? op.args[3]._op.val : op.args[1]._op.val
    kw_dict = kwargs2dict(op)
    dim = get(kw_dict, :dim, ndims(data))
    axis = ndims(data) - dim
    nd = NodeProto("Gather", op, Dict(:axis => axis))
    push!(g.node, nd)
end


##############################################################################
#                                    API                                     #
##############################################################################

"""
    save(io::IO, tape::Ghost.Tape{ONNXCtx})
    save(filename::String, tape::Ghost.Tape{ONNXCtx})

Save tape as an ONNX model. The way a particular operation is serialized is
controlled by methods of [save_node!](@ref).

See also: [`load!`](@ref)
"""
function save(io::IO, tape::Tape{ONNXCtx})
    g = graphproto()
    g.name = "generated_model"
    for (i, op) in enumerate(tape)
        if op isa Ghost.Input
            # add input to g.input, but not to g.initializer
            push!(g.input, ValueInfoProto(op))
        elseif op isa Ghost.Constant
            # add constant to g.initializer, but not to g.input
            # some models out there also put constants & parameters
            # to g.init, but it seems to be an outdated practise
            push!(g.initializer, TensorProto(op.val, onnx_name(op)))
        elseif op isa Ghost.Call
            save_node!(g, op)
        else
            error("$(typeof(op)) is not yet supported in model export")
        end
    end
    res = tape[tape.result]
    if res.val isa Tuple
        # if the last operation in the graph is multi-output, there must be
        # unpacked elements of that var
        vars = unpacked_vars(res)
        @assert(all(v isa V for v in vars), "Cannot save the tape because the result " *
            "is multi-output, but its elements aren't destructured to the tape")
        for v in vars
            push!(g.output, ValueInfoProto(tape[v]))
        end
    else
        push!(g.output, ValueInfoProto(tape[tape.result]))
    end
    m = modelproto();
    m.graph = g;
    writeproto(io, m)
end


function save(filename::String, tape::Tape{ONNXCtx})
    open(filename, "w") do io
        save(io, tape)
    end
end