import os
from ultralytics import YOLO
from pathlib import Path

import config

for task in ["det"]:  # ["det", "seg", "pose"]
    task_dir = Path(config.ROOT) / "models" / task
    task_dir.mkdir(parents=True, exist_ok=True)

    for size in ["n"]: # ["n", "s", "m", "l", "x"]
        pt_name = f"yolo11{size}.pt" if task == "det" else f"yolo11{size}-{task}.pt"
        pt_path = task_dir / pt_name
        model = YOLO(str(pt_path))