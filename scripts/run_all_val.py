import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import config

ROOT = Path(config.ROOT)
MODEL_ROOT = ROOT / "quantized_models"
RUNS_ROOT = ROOT / "runs"
DATASET = "coco128"

def get_task(model_name: str) -> str:
    if "-seg" in model_name:
        return "segment"
    elif "-pose" in model_name:
        return "pose"
    return "detect"

def run_val(model: Path) -> str | None:
    rel_path = model.relative_to(ROOT)
    rel_dir = rel_path.parent
    model_name = model.stem
    task = get_task(model_name)

    out_dir = RUNS_ROOT / rel_dir / model_name / DATASET
    if (out_dir / "val.log").exists():
        print(f"Skipping: {rel_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yolo", "val",
        f"model={model}",
        f"data={DATASET}.yaml",
        f"task={task}",
        f"name={DATASET}",
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
        models = sorted(
            set(MODEL_ROOT.rglob("*.pt")) | set(MODEL_ROOT.rglob("*.onnx"))
        )
    else:
        print(f"Error: {MODEL_ROOT} is invalid")
        sys.exit(1)

    print(f"Validating {len(models)} models in parallel")

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {executor.submit(run_val, model): model for model in models}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"  Completed: {i}/{len(models)} — {result}", flush=True)

    print(f"\nRuns saved to: {RUNS_ROOT}")