# How to contribute

The easiest way to contribute to this package is to add a new [ONNX operator](https://github.com/onnx/onnx/blob/main/docs/Operators.md). To do so, one needs to:

1. Add a new method to `load_node()`.
2. Add a new method to `save_node!()`.
3. Write tests.


## Adding a new method to `load_node()`

`load_node()` loads a single ONNX operator from a graph onto a `Ghost.Tape`. It has the following signture:

```julia
load_node!(tape::Tape, ::OpConfig{BE, Op}, args::VarVec, attrs::AttrDict)
```

Where:

* `Ghost.Tape` represents computational graph in Julia
* `OpConfig{BE, Op}` is used for dispatching on backend `BE` and operator `Op`
* `VarVec` (alias to `Vector{Ghost.Variable}`) is a list of input variables to this operator
* `AttrDict` (alias to `Dict{Symbol, Any}`) is a dict of ONNX operator attributes


Let's see an example:

```julia
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Relu}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, NNlib.relu, args[1])
end
```

Here we translate [Relu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu) operator to a single call to `NNlib.relu`. Both implementations - in ONNX and in NNlib - take a single argument, which we pass to the call. Note that `args[1]` refers to a [variable](https://dfdx.github.io/Ghost.jl/dev/tape/#Variables) on the tape in Julia (column-major) format.

A more involved example is [Gemm](https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm) operator:

```julia
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
```

Here we have several complications. First, we split logic into 2 paths: we translate simple cases to just `*` and for more complex cases we implement our own `onnx_gemm` function. Second, in the `onnx_gemm` case we also translate ONNX attributes into function's keyword arguments. Third, in the `*` case we swap the arguments. This is somewhat unusual thing, which we need to account for the difference between ONNX's row-major and Julia's column-major arrays: ONNX.jl automatically reverses dimensions of parameter arrays when reading from the `.onnx` files and maintains Julia-friendly ordering during the loading, but some adjustments in operators may still be needed. Of course, such cases must be well thought out and thoroughly tested.

Find more examples in [save.jl](src/save.jl)

## Adding a new method to `save_node!()`

`save_node!()` is the opposite of `load_node()`.`save_node` takes a `Ghost.Call` and adds the corresponding operator(s) to the ONNX graph. Its signature looks like this:

```julia
save_node!(g::GraphProto, ::OpConfig{BE, Fn}, op::Ghost.Call)
```

Where:

* `GraphProto` is ONNX's data structure representing actual computational graph
* `OpConfig{BE, Fn}` is used for dispatching on beckend `BE` and Julia function type `Fn`
* `Ghost.Call` represents a single call to `f::Fn` on a `Tape`

Example:

```julia
function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(relu)}, op::Ghost.Call)
    nd = NodeProto("Relu", op)
    push!(g.node, nd)
end
```

`NodeProto(op_type::String, op::Ghost.Call, attrs::Dict=Dict())` is a convenient constructor that creates an ONNX node of the provided type and maps Julia function arguments (of type `Ghost.Variable`) to names of the corresponding arguments in the already built ONNX graph. This is enough for a large portion of operators.

Let's now see a more example:

```julia
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
```

In the `load_node()` above we reversed the order of the arguments. When saving the node, we must do the same thing. Thus, we need to construct a `NodeProto` manually. `onnx_name(v)` generates a valid ONNX name from a variable. The rest of the code should be self-explanatory.

Here's also `save_node!()` for `onnx_gemm` version:

```julia
function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_gather), op::Ghost.Call)
    data = iskwfunc(op.fn) ? op.args[3]._op.val : op.args[1]._op.val
    kw_dict = kwargs2dict(op)
    dim = get(kw_dict, :dim, ndims(data))
    axis = ndims(data) - dim
    nd = NodeProto("Gather", op, Dict(:axis => axis))
    push!(g.node, nd)
end
```

Note that in this snippet instead if `OpConfig{...}` type we used `@opconfig_kw(...)` macro. This macros is expanded into a definition that catches both - normal and kw version of a function:

```julia
OpConfig{:ONNX, <:Union{typeof(onnx_gemm), typeof(Core.kwfunc(onnx_gemm))}}
```

More examples can be found in [save.jl](src/save.jl).

## Testing

`ort_test()` takes a Julia function, creates a `Tape` and saves it as an `.onnx` file, then uses `ONNXRunTime.jl` to run it and finally loads it back. The usage is as simple as this:

```julia
x = rand(3, 4)
ort_test(ONNX.relu, x)
```