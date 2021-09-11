function Base.show(io::IO, nd::NodeProto)
    out = (nd.output...,)
    inp = (nd.input...,)
    println(io, "$out = $(nd.op_type)$inp")
    for (k, v) in Dict(nd.attribute)
        println(io, "    $k => $v")
    end
end

function Base.show(io::IO, g::GraphProto)
    for nd in g.node
        show(io, nd)
    end
end