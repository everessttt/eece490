from pathlib import Path
import torch
from ultralytics import YOLO

import config
from quantization_helpers import quantization_methods
from PoTPTQ import PoTPTQ_2, get_calibration_images


def quantize_pytorch_model(yolo_model, quantization_fn, num_bits: int):
    model: torch.nn.Module = yolo_model.model # type: ignore
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dtype != torch.float32 or param.ndim < 2:
                continue
            param.copy_(quantization_fn(param.data, num_bits=num_bits))


if __name__ == "__main__":
    model_root = Path(config.ROOT) / "models"
    output_root = Path(config.ROOT) / "quantized_models"

    pt_files = list(model_root.rglob("*.pt"))
    total = len(pt_files) * len(quantization_methods) * 31
    completed = 0

    calib_imgs = get_calibration_images(n=128)

    for pt_path in pt_files:
        model_name = pt_path.stem
        relative_dir = pt_path.parent.relative_to(model_root)

        for quant_fn in quantization_methods:
            for bits in range(2, 33):
                out_path = (
                    output_root
                    / quant_fn.__name__
                    / relative_dir
                    / model_name
                    / f"{model_name}_{bits}bit.onnx"
                )
                if out_path.exists():
                    completed += 1
                    print(f"\r{completed}/{total}", end="", flush=True)
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    yolo = YOLO(str(pt_path))
                    quantize_pytorch_model(yolo, quant_fn, bits)

                    if quant_fn.__name__ == "PoTPTQ" and bits <= 3:
                        PoTPTQ_2(
                            yolo.model, # type: ignore
                            calib_imgs,
                            num_bits=bits,
                            epochs=10 if bits >= 3 else 40,
                            lr=1e-3,
                        )

                    exported = Path(yolo.export(format="onnx", dynamic=True, simplify=True))
                    exported.rename(out_path)

                except Exception as e:
                    print(f"\nSkipped {out_path}: {e}")

                completed += 1
                print(f"\r{completed}/{total}", end="", flush=True)

    print(f"\nQuantized models saved to: {output_root}")