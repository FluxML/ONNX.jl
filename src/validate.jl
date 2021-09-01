# copy from https://github.com/DrChainsaw/ONNXNaiveNASflux.jl/blob/26c7bbb4977d310dd67df0ae884bfaac3f78216a/src/validate.jl

"""
    validate(mp::ModelProto)
    validate(mp::ModelProto, fs...)
Validate `mp`, throwing an exception if it is invalid.
It is possible to specify the validation steps `fs` to perform. Default is `uniqueoutput, optypedefined, outputused, inputused, hasname`
"""
validate(mp::ModelProto, fs...=(uniqueoutput, optypedefined, outputused, inputused, hasname)...) = foreach(f -> f(mp), fs)

errinfo(n::NodeProto) = "NodeProto with: \n"  * join(["\t$pn:\t$(clstring(getproperty(n, pn)))" for pn in propertynames(n) if hasproperty(n, pn)], "\n")
clstring(x::AbstractArray) = "[" * join(string.(x), ", ") * "]"
clstring(x) = string(x)

"""
    uniqueoutput(mp::ModelProto, or=error)
    uniqueoutput(gp::GraphProto, or=error)
Test that output names are unique. If not, an error message will be passed to `or`.
"""
uniqueoutput(mp::ModelProto, or=error) = uniqueoutput(mp.graph, or)
function uniqueoutput(gp::GraphProto, or=error)
    d = Dict()
    for n in gp.node
        for oname in n.output
            if haskey(d, oname)
                or("Duplicate output name: $oname found in \n$(errinfo(d[oname])) \nand\n $(errinfo(n))")
            end
            d[oname] = n
        end
    end
end

"""
    optypedefined(mp::ModelProto, or=error)
    optypedefined(gp::GraphProto, or=error)
Test that operations are defined for each node. If not, an error message will be passed to `or`.
"""
optypedefined(mp::ModelProto, or=error) = optypedefined(mp.graph, or)
function optypedefined(gp::GraphProto, or=error)
    for n in gp.node
        hasproperty(n, :op_type) || or("No op_type defined for $(errinfo(n))")
    end
end

"""
    outputused(mp::ModelProto, or=error)
    outputused(gp::GraphProto, or=error)
Test that all outputs are used. If not, an error message will be passed to `or`.
"""
outputused(mp::ModelProto, or=error) = outputused(mp.graph, or)
function outputused(gp::GraphProto, or=error)
    found, used = ioused(gp)
    unusedouts = setdiff(found, used)
    str(s) = join(sort(collect(s)), ", ")
    isempty(unusedouts) || or("Found unused outputs: $(str(unusedouts))")
end

"""
    inputused(mp::ModelProto, or=error)
    inputused(gp::GraphProto, or=error)
Test that all inputs are used. If not, an error message will be passed to `or`.
"""
inputused(mp::ModelProto, or=error) = inputused(mp.graph, or)
function inputused(gp::GraphProto, or=error)
    used, found = ioused(gp)
    unusedins = setdiff(found, used)
    str(s) = join(sort(collect(s)), ", ")
    isempty(unusedins) || or("Found unused inputs: $(str(unusedins))")
end

name(x::TensorProto) = x.name
name(x::ValueInfoProto) = x.name

function ioused(gp::GraphProto)
    found = union(Set(name.(gp.input)), Set(name.(gp.initializer)))
    used = Set(name.(gp.output))
    for n in gp.node
        foreach(oname -> push!(found, oname), n.output)
        foreach(iname -> push!(used, iname), n.input)
    end

    return found, used
end

hasname(mp::ModelProto, or=error) = hasname(mp.graph, or)
function hasname(gp::GraphProto, or=error)
     hasproperty(gp, :name) || return or("Graph name not defined!")
     isempty(gp.name) && or("Graph name is empty string!")
 end
