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

This packages is currently under development, don't tell us we didn't warn you!