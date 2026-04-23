import os
import onnx
from onnx import numpy_helper
import numpy as np
from pathlib import Path
from typing import Union
from multiprocessing import Pool, Value

from quantization_helpers import range_methods, quantization_methods

counter = None
def init_counter(c):
    global counter
    counter = c

def quantize_model(
    onnx_model_path: Union[str, Path],
    quantized_model_path: Union[str, Path],
    quantization_fn,
    range_fn,
    num_bits: int
):
    onnx_model_path = Path(onnx_model_path)
    quantized_model_path = Path(quantized_model_path)
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_model_path}")

    model = onnx.load(str(onnx_model_path))
    quantized_count = 0

    for initializer in model.graph.initializer:
        tensor = numpy_helper.to_array(initializer)
        if tensor.dtype != np.float32 or tensor.size == 0:
            continue
        new_tensor = quantization_fn(tensor, num_bits=num_bits, range_fn=range_fn)
        initializer.CopyFrom(numpy_helper.from_array(new_tensor, name=initializer.name))
        quantized_count += 1

    quantized_model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(quantized_model_path))

def quantize_task(args):
    onnx_model_path, output_path, quantization_fn, range_fn, bits = args
    model_name = Path(onnx_model_path).stem
    out_path = Path(output_path) / quantization_fn.__name__ / range_fn.__name__ / f"{model_name}_{bits}bit.onnx"
    if not out_path.exists():
        quantize_model(
            onnx_model_path=onnx_model_path,
            quantized_model_path=out_path,
            quantization_fn=quantization_fn,
            range_fn=range_fn,
            num_bits=bits,
        )
    with counter.get_lock():
        counter.value += 1
        print(f"  Completed: {counter.value}/{counter.total}", end="\r", flush=True)

if __name__ == "__main__":
    ONNX_MODEL_ROOT = Path("/home/everest/EECE490/models")
    OUTPUT_ROOT = Path("/home/everest/EECE490/quantized_models")

    onnx_models = list(ONNX_MODEL_ROOT.rglob("*.onnx"))

    tasks = [
        (onnx_path, OUTPUT_ROOT / onnx_path.relative_to(ONNX_MODEL_ROOT).parent / onnx_path.stem, quantization_fn, range_fn, bits,)
        for onnx_path in onnx_models
        for bits in range(2, 33)
        for quantization_fn in quantization_methods
        for range_fn in range_methods
    ]

    total = len(tasks)
    counter = Value("i", 0)
    counter.total = total

    print(f"Total tasks: {total}")
    with Pool(processes=os.cpu_count(), initializer=init_counter, initargs=(counter,)) as pool:
        pool.map(quantize_task, tasks)

    print(f"\nModels saved to: {OUTPUT_ROOT}")