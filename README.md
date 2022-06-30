# ONNX

ONNX.jl is in the process of a total reconstruction and currently supports saving & loading graphs as a [`Ghost.Tape`](https://dfdx.github.io/Ghost.jl/dev/reference/#Ghost.Tape). When possible, functions from `NNlib` or standard library are used, but no conversion to Flux is implemented yet. See [resnet18.jl](examples/resnet18.jl) for a practical example of graph loading.

Not all ONNX operators are implemented. See [How to contribute](CONTRIBUTE.md) for details of adding new operators.