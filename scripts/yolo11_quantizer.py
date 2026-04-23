"""
Quantize yolo11n.onnx using onnxruntime static quantization.
Produces a QDQ-format ONNX with int8 activations/weights.
Compatible with STEdgeAI/STM32N6 deployment.

Output: /home/everest/EECE490/quantized_models/yolo11n_qdq_int8.onnx
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
    shape_inference,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "/home/everest/EECE490/yolo11n.onnx"
OUTPUT_PATH      = "/home/everest/EECE490/quantized_models/yolo11n_qdq_int8.onnx"
CALIB_IMAGE_DIR  = "/home/everest/EECE490/val2017"
INPUT_SIZE       = (256, 256)   # (H, W)
MAX_CALIB_IMAGES = 200
# ──────────────────────────────────────────────────────────────────────────────


def preprocess_image(image_path):
    """Letterbox resize + normalize to [0,1] float32 in CHW format."""
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    target_w, target_h = INPUT_SIZE[1], INPUT_SIZE[0]
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    img = img.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new("RGB", (target_w, target_h), (127, 127, 127))
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]


class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, max_images, input_name):
        self.input_name = input_name
        self.data = []

        image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:max_images]

        print(f"[INFO] Loading {len(image_files)} calibration images...")
        for i, path in enumerate(image_files):
            try:
                self.data.append({self.input_name: preprocess_image(path)})
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(image_files)} loaded")

        print(f"[INFO] Calibration ready: {len(self.data)} images")
        self._index = 0

    def get_next(self):
        if self._index >= len(self.data):
            return None
        item = self.data[self._index]
        self._index += 1
        return item


def fix_uint8_to_int8(model):
    """Convert all uint8 zero points to int8 for STEdgeAI compatibility."""
    fixed = 0
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.UINT8:
            arr = numpy_helper.to_array(init).astype(np.int32)
            arr_int8 = (arr - 128).astype(np.int8)
            new_init = numpy_helper.from_array(arr_int8, name=init.name)
            init.CopyFrom(new_init)
            fixed += 1

    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
            vi.type.tensor_type.elem_type = onnx.TensorProto.INT8
            fixed += 1

    print(f"[INFO] Fixed {fixed} uint8 tensors -> int8")
    return model


def get_input_name(model_path):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)
    return sess.get_inputs()[0].name


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Step 1: Shape inference preprocessing
    print("[STEP 1] Running shape inference preprocessing...")
    preprocessed_path = MODEL_PATH.replace(".onnx", "_preprocessed.onnx")
    shape_inference.quant_pre_process(
        input_model_path=MODEL_PATH,
        output_model_path=preprocessed_path,
        skip_optimization=False,
    )

    # Step 2: Static quantization
    input_name = get_input_name(preprocessed_path)
    print(f"[INFO] Input name: '{input_name}'")

    reader = YoloCalibrationDataReader(
        image_dir=CALIB_IMAGE_DIR,
        max_images=MAX_CALIB_IMAGES,
        input_name=input_name,
    )

    tmp_output = OUTPUT_PATH.replace(".onnx", "_tmp.onnx")
    print("[STEP 2] Running static quantization (QDQ, per-channel, int8)...")
    quantize_static(
        model_input=preprocessed_path,
        model_output=tmp_output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    # Step 3: Fix any remaining uint8 zero points
    print("[STEP 3] Fixing uint8 zero points -> int8 for STEdgeAI compatibility...")
    model = onnx.load(tmp_output)
    model = fix_uint8_to_int8(model)
    onnx.save(model, OUTPUT_PATH)

    # Cleanup temp files
    os.remove(tmp_output)
    os.remove(preprocessed_path)

    orig_mb  = os.path.getsize(MODEL_PATH) / 1024 / 1024
    quant_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n[DONE] Quantized model: {OUTPUT_PATH}")
    print(f"  Original:  {orig_mb:.2f} MB")
    print(f"  Quantized: {quant_mb:.2f} MB  ({100*quant_mb/orig_mb:.1f}% of original)")


if __name__ == "__main__":
    main()