function Base.show(io::IO, nd::NodeProto)
    out = (nd.output...,)
    inp = (nd.input...,)
    println(io, "$out = $(nd.op_type)$inp")
    if nd.op_type != "Constant"  # constants are too verbose
        for (k, v) in Dict(nd.attribute)
            println(io, "    $k => $v")
        end
    end
end

function Base.show(io::IO, g::GraphProto)
    for nd in g.node
        show(io, nd)
    end
end