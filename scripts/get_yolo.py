from ultralytics import YOLO
from pathlib import Path
import torchvision
import torch

import config

for task in ["det", "seg", "pose"]:  # ["det", "seg", "pose"]
    task_dir = Path(config.ROOT) / "models" / task
    task_dir.mkdir(parents=True, exist_ok=True)

    for size in ["n", "s", "m", "l", "x"]: # ["n", "s", "m", "l", "x"]
        pt_name = f"yolo11{size}.pt" if task == "det" else f"yolo11{size}-{task}.pt"
        pt_path = task_dir / pt_name
        model = YOLO(str(pt_path))

# Faster R-CNN
# det_dir = Path(config.ROOT) / "models" / "det"
# det_dir.mkdir(parents=True, exist_ok=True)

# fasterrcnn_path = det_dir / "fasterrcnn_resnet50_fpn_coco.pt"
# if not fasterrcnn_path.exists():
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
#     torch.save(model.state_dict(), fasterrcnn_path)

# # EfficientDet
# efficientdet_path = det_dir / "efficientdet_d0.pt"
# if not efficientdet_path.exists():
#     from effdet import create_model
#     model = create_model("tf_efficientdet_d0", pretrained=True)
#     torch.save(model.state_dict(), efficientdet_path)