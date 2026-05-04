import subprocess
from pathlib import Path
import sys
import config

ROOT = Path(config.ROOT)
MODEL_ROOT = ROOT / "models"
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
    name = model_name.split(".")[0]
    size = name.split("_")[0]
    return size[-1] if size else "null"

def run_val(model: Path) -> str | None:
    rel_path = model.relative_to(ROOT)
    rel_dir = rel_path.parent
    model_name = model.stem
    model_size = get_model_size(model_name)
    task = get_task(model_name)

    dataset = get_dataset(model_name)
    dataset_name = get_dataset_name(model_name)
    out_dir = RUNS_ROOT / rel_dir / model_name / dataset_name

    if not (out_dir / "val.log").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["yolo", "val", f"model={model}", f"data={dataset}", f"task={task}", f"name={dataset_name}", f"project={RUNS_ROOT / rel_dir / model_name}", "half=False", "save_json=True", "save_conf=True", "save_txt=True", "plots=True", "verbose=False", "exist_ok=True"]
        with open(out_dir / "val.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

    # if task == "detect" and model_size == "n":
    #     out_dir_coco = RUNS_ROOT / rel_dir / model_name / "coco"
    #     if not (out_dir_coco / "val.log").exists():
    #         out_dir_coco.mkdir(parents=True, exist_ok=True)
    #         cmd_coco = ["yolo", "val", f"model={model}", "data=coco.yaml", f"task={task}", "name=coco", f"project={RUNS_ROOT / rel_dir / model_name}", "half=False", "save_json=True", "save_conf=True", "save_txt=True", "plots=True", "verbose=False", "exist_ok=True"]
    #         with open(out_dir_coco / "val.log", "w") as log:
    #             subprocess.run(cmd_coco, stdout=log, stderr=subprocess.STDOUT, text=True)

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