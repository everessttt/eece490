from ultralytics import YOLO
from pathlib import Path

import config

for task in ["det", "seg", "pose"]:
    for arch in ["yolo11", "yolo26"]:
        task_dir = Path(config.ROOT) / "models" / arch / task
        task_dir.mkdir(parents=True, exist_ok=True)

        for size in ["n", "s", "m", "l", "x"]:
            pt_name = f"{arch}{size}.pt" if task == "det" else f"{arch}{size}-{task}.pt"
            model = YOLO(str(task_dir / pt_name))

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