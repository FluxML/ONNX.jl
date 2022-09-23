using ONNX
using Images
import Umlaut: Tape, play!


include("imagenet_classes.jl")


function imread(path::AbstractString; sz=nothing)
    img = Images.load(path);
    if sz !== nothing
        img = imresize(img, sz);
    end
    x = convert(Array{Float32}, channelview(img))
    # CHW -> WHC
    x = permutedims(x, (3, 2, 1))
    return x
end


function maxk(a, k)
    b = partialsortperm(a, 1:k, rev=true)
    return collect(zip(b, a[b]))
end


function test_image(tape::Tape, path::AbstractString)
    x = imread(expanduser(path); sz=(224, 224))
    x = reshape(x, size(x)..., 1)
    y = play!(tape, x)
    y = reshape(y, size(y, 1))
    top = maxk(y, 10)
    for (i, (idx, val)) in enumerate(top)
        name = IMAGENET_CLASSES[idx - 1]
        println("$i: $name ($val)")
    end
end


function main()
    path = "resnet18.onnx"
    if !isfile(path)
        download("https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx", path)
    end
    # dummy input
    img = rand(Float32, 224, 224, 3, 1)
    # load the model as a Umlaut.Tape
    println("Loading the model")
    resnet = ONNX.load(path, img)
    # test a few images
    println("Image of a guitar:")
    guitar_path = download("https://cdn.pixabay.com/photo/2015/05/07/11/02/guitar-756326_960_720.jpg")
    test_image(resnet, guitar_path)

    println("\nImage of a goose:")
    goose_path = download("https://upload.wikimedia.org/wikipedia/commons/3/3f/Snow_goose_2.jpg")
    test_image(resnet, goose_path)
end

main()
