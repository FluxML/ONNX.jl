import sys
import os
import onnx
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def check_and_run(path):
    path = os.path.expanduser(path)

    print("Checking the model... ", end="")
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print("Ok")

    print("Running the model")
    ort_session = onnxruntime.InferenceSession(path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)


if __name__ == "__main__":
    # path =  "~/data/onnx/generated_model.onnx"
    path = sys.argv[1]
    check_and_run(path)


