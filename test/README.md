## Model Tests:

While node operator tests are useful in testing a specific operator, Model 
tests are used to test the model as a whole. These models can be pretty large (several 
hundered MBs at times), and hence individual models are downloaded as and when they need 
to be tested. Model tests are used not only to verify the functioning of the 
operators working as a single unit (every operator taking an input from anouther node 
and feeding the output to another), but these models can also be used directly for any 
task, without having to reinvent the wheel by building and training the model from scratch.

## Running model tests

You need to run the `modeltests.jl` script to run the model tests on a specific model.

For example, to test the MNIST pretrained model, run:

```
julia modeltests.jl MNIST
```

This creates a new `models` directory, downloads and extracts the MNIST pre-trained model
and tests it on the test data provided. (Note: You need to have `wget` installed to 
download the model, and `tar` installed to extract it.)

Currently, four model tests are available. These include the MNIST, Squeezenet, VGG19 and 
Emotion_Ferplus models.

## Writing your own node tests

As these models become more diverse, it is likely that you might come across operators that
aren't supported by ONNX.jl. In such as case, you might have to implement it yourself (feel 
free to open an issue too).

The `ops.jl` file (`src/graph/ops.jl`) contains the implementation of all operators at this 
point. In order to test your implementation, you need to make sure that ONNX provides the 
test data for the operator. The `main_test` is the main function used to test individual 
operators. It takes in the name of the test file, the expected output and the inputs as its
arguments. Also, please do create a Pull Request if your tests pass, as it might be useful 
for the community.

