import sys
import os
import onnx
import onnxruntime
import numpy as np


def check_and_run(path):
    path = os.path.expanduser(path)

    print("Checking the model... ", end="")
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print("Ok")

    print("Running the model")
    ort_session = onnxruntime.InferenceSession(path)
    # compute ONNX Runtime output prediction
    ort_inputs = {
        ort_session.get_inputs()[0].name: np.random.rand(4, 3),
        ort_session.get_inputs()[1].name: np.random.rand(4, 3)
    }
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)


if __name__ == "__main__":
    # path =  "~/data/onnx/generated_model.onnx"
    path = sys.argv[1]
    check_and_run(path)


