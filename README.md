# ONNX

[![Build Status](https://travis-ci.org/ayush1999/ONNX.jl.svg?branch=master)](https://travis-ci.org/ayush1999/ONNX.jl.svg?branch=master)


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

It's always better to run the tests before moving on to importing a model. The operator tests ensure that all ops are working. Use `]test ONNX` to run the tests.

* Running these tests may take some time, as it may initially download the test files if you don't already have them.(You need to have git preinstalled in order to download the tests)

In order to read more about these tests and run model specific tests, please go through the docs in the `test` directory. 

## Contributing and Help

If you're looking to contribute to the development of this package, and don't know where to begin, [this blog post](https://medium.com/@ayush1999/onnx-jl-the-past-present-and-future-d3b497a0cd4c) can be a good 
starting point. It lists the approach taken towards developing this package, the current obstacles, and the work to be done in the future.

Since this package is currently under development, feel free to open an [issue](https://github.com/FluxML/ONNX.jl/issues) if you find any error/bug. 

For more discussion, you can get in touch with us on [Julia Slack](https://slackinvite.julialang.org/). We're pretty active on the #machine-learning channel.