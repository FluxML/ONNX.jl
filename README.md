# ONNX

[![Build Status](https://travis-ci.org/MikeInnes/ONNX.jl.svg?branch=master)](https://travis-ci.org/MikeInnes/ONNX.jl)

[![Coverage Status](https://coveralls.io/repos/MikeInnes/ONNX.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/MikeInnes/ONNX.jl?branch=master)

[![codecov.io](http://codecov.io/github/MikeInnes/ONNX.jl/coverage.svg?branch=master)](http://codecov.io/github/MikeInnes/ONNX.jl?branch=master)


ONNX.jl : Read [ONNX](https://onnx.ai/) graphs and load these models in Julia. ONNX.jl provides an instance of transfer learning into Julia, by reading pretrained models from ONNX format to [Flux.jl](https://github.com/FluxML/Flux.jl). This is done by generating the DataFlow graph from the model, and then reading it as Julia code.

## Loading models

You need to have the `model.onnx` ( or in some cases `model.pb` ) file,  which will be read. Several pretrained ONNX model files can also be downloaded from [here](https://github.com/onnx/models).  Now that we have the `model.onnx` file, we can read it into Flux as :

```
julia> using Flux, ONNX                             # Import the required packages.
julia> ONNX.load_model("model.onnx")                # If you are in some other directory, specify the entire path.
                                                    # This creates two files: model.jl and weights.bson.
julia> weights = ONNX.load_weights("weights.bson")  # Read the weights from the binary serialized file.
julia> model = include("model.jl")                  # Loads the model from the model.jl file.
```

And `model` is the corresponding model in Flux!

This package is currently under development, don't tell us we didn't warn you!

## Running the tests

It's always better to run the tests before moving on to importing a model. The operator tests ensure that all ops are working. Follow the given steps to run these tests:

* Change your working directory to the `test` directory (`cd ~/.julia/v0.6/ONNX.jl/test` from the terminal)

* Inside the test repository, run the `runtests.jl` script. (`julia runtests.jl`).

* Running these tests may take some time, as it may initially download the test files if you don't already have them.(You need to have git preinstalled in order to download the tests)


## Contributing and Help

Since this package is currently under development, feel free to open an [issue](https://github.com/FluxML/ONNX.jl/issues) if you find any error/bug. 

For more discussion, you can get in touch with us on [Julia Slack](https://slackinvite.julialang.org/). We're pretty active on the #machine-learning channel.