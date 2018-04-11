# ONNX

[![Build Status](https://travis-ci.org/MikeInnes/ONNX.jl.svg?branch=master)](https://travis-ci.org/MikeInnes/ONNX.jl)

[![Coverage Status](https://coveralls.io/repos/MikeInnes/ONNX.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/MikeInnes/ONNX.jl?branch=master)

[![codecov.io](http://codecov.io/github/MikeInnes/ONNX.jl/coverage.svg?branch=master)](http://codecov.io/github/MikeInnes/ONNX.jl?branch=master)


ONNX.jl : Read [ONNX](https://onnx.ai/) graphs and load these models in Julia. ONNX.jl provides an instance of transfer learning into Julia, by reading pretrained models from ONNX format to [Flux.jl](https://github.com/FluxML/Flux.jl). This is done by generating the DataFlow graph from the model, and then reading it as Julia code.

## Loading models

You need to have the `model.onnx` ( or in some cases `model.pb` ) file,  which will be read. Several pretrained ONNX model files can also be downloaded from [here](https://github.com/onnx/models).  Now that we have the `model.onnx` file, we can read it into Flux as :

```
julia> using Flux, ONNX                             # Import the required packages.
julia> weights, model_expr = ONNX.load_model("model.onnx")                # If you are in some other directory, specify the entire path.
                                                    # This creates two files: model.jl and weights.bson.
julia> model = eval(model_expr)                     # eval the model expression into the current module

```
Or alternatively, use `include` to the load the model from the model.jl file:
`julia> model = include("model.jl")`


And `model` is the corresponding model in Flux!

This package is currently under development, don't tell us we didn't warn you!

## Contributing and Help

Since this package is currently under development, feel free to open an [issue](https://github.com/FluxML/ONNX.jl/issues) if you find any error/bug. 

For more discussion, you can get in touch with us on [Julia Slack](https://slackinvite.julialang.org/). We're pretty active on the #machine-learning channel.
