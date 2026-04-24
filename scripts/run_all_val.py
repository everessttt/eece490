import subprocess
from pathlib import Path
import sys
import config

ROOT = Path(config.ROOT)
MODEL_ROOT = ROOT / "quantized_models"
RUNS_ROOT = ROOT / "runs"

def get_task(model_name: str) -> str:
    if "-seg" in model_name:
        return "segment"
    elif "-pose" in model_name:
        return "pose"
    return "detect"

def get_dataset(model_name: str) -> str:
    if "-seg" in model_name:
        return "coco8-seg.yaml"
    elif "-pose" in model_name:
        return "coco8-pose.yaml"
    return "coco128.yaml"

def get_dataset_name(model_name: str) -> str:
    if "-seg" in model_name:
        return "coco8-seg"
    elif "-pose" in model_name:
        return "coco8-pose"
    return "coco128"

def get_model_size(model_name: str) -> str:
    base = model_name.split("-")[0]
    return base[-1]

def run_val(model: Path) -> str | None:
    rel_path = model.relative_to(ROOT)
    rel_dir = rel_path.parent
    model_name = model.stem
    model_size = get_model_size(model_name)
    dataset_name = get_dataset_name(model_name)
    task = get_task(model_name)

    out_dir = RUNS_ROOT / rel_dir / model_name / dataset_name
    if(out_dir / "val.log").exists():
        return
    if("-seg" in model_name or "-pose" in model_name) and model_size in ["s", "m", "l", "x"]:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    data = get_dataset(model_name)
    if task == "detect" and get_model_size(model_name) == "n":
        data = "coco.yaml"

    cmd = [
        "yolo",
        "val",
        f"model={model}",
        f"data={data}",
        f"task={task}",
        f"name={dataset_name}",
        f"project={RUNS_ROOT / rel_dir / model_name}",
        "half=False",
        "save_json=True",
        "save_conf=True",
        "save_txt=True",
        "plots=True",
        "verbose=False",
        "exist_ok=True",
    ]

    with open(out_dir / "val.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

    return str(rel_path)

if __name__ == "__main__":
    if MODEL_ROOT.is_file():
        models = [MODEL_ROOT]
    elif MODEL_ROOT.is_dir():
        models = sorted(set(MODEL_ROOT.rglob("*.pt")) | set(MODEL_ROOT.rglob("*.onnx")))
    else:
        print(f"Error: {MODEL_ROOT} is invalid")
        sys.exit(1)

    total = len(models)
    print(f"Validating {total} models\n")
    for i, model in enumerate(models, 1):
        result = run_val(model)
        status = result if result else "(skipped)"
        print(f"  {i}/{total} — {status}", flush=True)

    print(f"Runs saved to: {RUNS_ROOT}\n")