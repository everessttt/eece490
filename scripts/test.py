import onnx
import numpy as np
from onnx import numpy_helper

orig = onnx.load(r"C:\github_projects\EECE490\models\det\yolo11n.onnx")
quant = onnx.load(r"C:\github_projects\EECE490\quantized_models\uniform_affine\det\yolo11n\yolo11n_32bit.onnx")

for o, q in zip(orig.graph.initializer, quant.graph.initializer):
    o_arr = numpy_helper.to_array(o)
    q_arr = numpy_helper.to_array(q)
    if not np.allclose(o_arr, q_arr, atol=1e-6):
        print(f"Mismatch: {o.name}, max diff: {np.max(np.abs(o_arr - q_arr)):.6f}")

print("Done")