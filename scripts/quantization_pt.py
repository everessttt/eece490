import torch
import numpy as np
from pathlib import Path
from typing import Union
from ultralytics import YOLO
import copy

from quantization_helpers import range_methods, quantization_methods, gptq

def get_calibration_inputs(model, calibration_images):
    calibration_inputs = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if name not in calibration_inputs:
                calibration_inputs[name] = []
            x = input[0].detach().cpu()
            x_unfolded = torch.nn.functional.unfold(x, kernel_size=module.kernel_size, padding=module.padding, stride=module.stride, dilation=module.dilation)
            calibration_inputs[name].append(x_unfolded.permute(1, 0, 2).reshape(x_unfolded.shape[1], -1).numpy())
        return hook

    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.groups == 1:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.model.eval()
    with torch.no_grad():
        for img in calibration_images:
            model(img, verbose=False)

    for hook in hooks:
        hook.remove()

    return {name: np.concatenate(inputs, axis=1) for name, inputs in calibration_inputs.items()}


def quantize_pt(
    pt_model_path: Union[str, Path],
    quantized_model_path: Union[str, Path],
    quantization_fn,
    range_fn,
    num_bits: int,
    model: YOLO = None,
    calibration_inputs: dict = None,
):
    pt_model_path = Path(pt_model_path)
    quantized_model_path = Path(quantized_model_path)
    if not pt_model_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_model_path}")

    if model is None:
        model = YOLO(str(pt_model_path))

    if quantization_fn is gptq:
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.groups == 1 and name in calibration_inputs:
                W = module.weight.data.cpu().numpy()
                out_ch, in_ch, kH, kW = W.shape
                W_2d = W.reshape(out_ch, in_ch * kH * kW)
                Q_2d = gptq(W_2d, num_bits=num_bits, calibration_input=calibration_inputs[name])
                module.weight.data = torch.from_numpy(Q_2d.reshape(out_ch, in_ch, kH, kW))
    else:
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                W = module.weight.data.cpu().numpy()
                module.weight.data = torch.from_numpy(
                    quantization_fn(W, num_bits=num_bits, range_fn=range_fn)
                )

    quantized_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(quantized_model_path))


if __name__ == "__main__":
    PT_MODEL_ROOT = Path("/home/everest/EECE490/models")
    OUTPUT_ROOT = Path("/home/everest/EECE490/quantized_models")
    CALIBRATION_DIR = Path("/home/everest/EECE490/datasets/datasets/coco128/images/train2017")

    calibration_images = list(CALIBRATION_DIR.glob("*.jpg"))[:128]
    pt_models = list(PT_MODEL_ROOT.rglob("*.pt"))

    tasks = [
        (pt_path, quantization_fn, range_fn, bits)
        for pt_path in pt_models
        for bits in range(2, 33)
        for quantization_fn in quantization_methods
        for range_fn in (range_methods if quantization_fn is not gptq else [None])
    ]

    # group tasks by model to avoid reloading
    tasks_by_model = {}
    for pt_path, quantization_fn, range_fn, bits in tasks:
        tasks_by_model.setdefault(pt_path, []).append((quantization_fn, range_fn, bits))

    total = len(tasks)
    completed = 0
    print(f"Total tasks: {total}")

    for pt_path, model_tasks in tasks_by_model.items():
        model = YOLO(str(pt_path))
        calibration_inputs = get_calibration_inputs(model, calibration_images)
        original_state = copy.deepcopy(model.model.state_dict())

        for quantization_fn, range_fn, bits in model_tasks:
            range_name = range_fn.__name__ if range_fn is not None else "na"
            out_path = (
                OUTPUT_ROOT
                / pt_path.relative_to(PT_MODEL_ROOT).parent
                / pt_path.stem
                / quantization_fn.__name__
                / range_name
                / f"{pt_path.stem}_{bits}bit.pt"
            )
            if not out_path.exists():
                # reset weights before each quantization
                model.model.load_state_dict(copy.deepcopy(original_state))
                quantize_pt(
                    pt_model_path=pt_path,
                    quantized_model_path=out_path,
                    quantization_fn=quantization_fn,
                    range_fn=range_fn,
                    num_bits=bits,
                    model=model,
                    calibration_inputs=calibration_inputs,
                )
            completed += 1
            print(f"  Completed: {completed}/{total}", end="\r", flush=True)

        del model, calibration_inputs, original_state

    print(f"\nModels saved to: {OUTPUT_ROOT}")