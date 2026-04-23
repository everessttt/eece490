import os
import subprocess
import shutil
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# coco
os.makedirs("coco", exist_ok=True)
for url in [
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]:
    name = url.split("/")[-1]
    subprocess.run(["wget", "-c", "-P", "coco", url], check=True)
    subprocess.run(["unzip", "-q", f"coco/{name}", "-d", "coco"], check=True)
    os.remove(f"coco/{name}")

# coco128
subprocess.run(["wget", "-c", "https://ultralytics.com/assets/coco128.zip"], check=True)
subprocess.run(["unzip", "-q", "coco128.zip"], check=True)
os.remove("coco128.zip")