import argparse
import cv2
import json
import subprocess
import time
import threading
import numpy as np
import dxcam
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = r"C:\github_projects\EECE490\quantized_models\uniform_affine\det\yolo11n\yolo11n_32bit.onnx"
WIN_NAME = Path(MODEL_PATH).name
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
TARGET_CAPTURE_FPS = 30
INIT_WIN_W = 1280
INIT_WIN_H = 720
RENDER_FPS_CAP = 60
RENDER_FRAME_TIME = 1.0 / RENDER_FPS_CAP
FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTURE_SOURCE = "camera"
CAMERA_INDEX = 0

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

def assign_color(cid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=int(cid) * 2654435761 & 0xFFFFFFFF)
    h = int(rng.integers(0, 180))
    hsv = np.array([[[h, 220, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def class_label(cid: int) -> str:
    return COCO_LABELS[cid] if cid < len(COCO_LABELS) else f"class_{cid}"

COLOUR_LUT = [assign_color(i) for i in range(max(len(COCO_LABELS), 128))]

latest_capture: np.ndarray | None = None
latest_result:  tuple | None = None
running = threading.Event()
running.set()
capture_lock = threading.Lock()
result_lock = threading.Lock()


def find_logitech_camera() -> int:
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-PnpDevice -Class Camera | Select-Object FriendlyName, Status | ConvertTo-Json"
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            devices = json.loads(result.stdout)
            if isinstance(devices, dict):
                devices = [devices]
            for i, dev in enumerate(devices):
                if "logitech" in dev.get("FriendlyName", "").lower() and dev.get("Status") == "OK":
                    return i
    except Exception:
        pass

    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return idx

    return 0


def letterbox(img: np.ndarray, target: tuple[int, int] = (640, 640)):
    h, w   = img.shape[:2]
    th, tw = target
    scale  = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((th, tw, 3), 114, dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy

def preprocess(frame: np.ndarray, input_shape: tuple[int, ...]):
    _, _, h, w = input_shape
    img, scale, dx, dy = letterbox(frame, (h, w))
    blob = img[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]
    return blob, scale, dx, dy

def postprocess(output: np.ndarray, frame_shape: tuple, scale: float, dx: int, dy: int) -> list[tuple[int, int, int, int, float, int]]:
    h_img, w_img = frame_shape[:2]
    if output.ndim == 3:
        output = output.squeeze(0)
    pred = output.T if output.shape[0] < output.shape[1] else output

    boxes_raw   = pred[:, :4]
    scores      = pred[:, 4:]
    class_ids   = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences >= CONF_THRESHOLD
    if not np.any(mask):
        return []

    boxes_raw   = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    cx, cy, bw, bh = boxes_raw.T
    x1 = np.clip((cx - bw / 2 - dx) / scale, 0, w_img)
    y1 = np.clip((cy - bh / 2 - dy) / scale, 0, h_img)
    x2 = np.clip((cx + bw / 2 - dx) / scale, 0, w_img)
    y2 = np.clip((cy + bh / 2 - dy) / scale, 0, h_img)

    valid = (x2 - x1 >= 1) & (y2 - y1 >= 1)
    if not np.any(valid):
        return []
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    confidences = confidences[valid]
    class_ids   = class_ids[valid]

    boxes_cv = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32).tolist()
    confs_cv = confidences.astype(np.float32).tolist()

    nms_result = cv2.dnn.NMSBoxes(boxes_cv, confs_cv, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(nms_result) == 0:
        return []

    idxs = np.array(nms_result).flatten()
    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), float(confidences[i]), int(class_ids[i])) for i in idxs]


def capture_worker():
    global latest_capture

    if CAPTURE_SOURCE == "desktop":
        camera = dxcam.create(output_idx=0)
        camera.start(target_fps=TARGET_CAPTURE_FPS, video_mode=True)

        while running.is_set():
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            with capture_lock:
                latest_capture = frame[:, :, ::-1].copy()

        camera.stop()

    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            running.clear()
            return

        cap.set(cv2.CAP_PROP_FPS, TARGET_CAPTURE_FPS)
        print("camera inference started")

        while running.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with capture_lock:
                latest_capture = frame.copy()

        cap.release()


def inference_worker(session: ort.InferenceSession, input_name: str, input_shape: tuple[int, ...]):
    global latest_result

    while running.is_set():
        with capture_lock:
            snapshot = latest_capture
        if snapshot is None:
            time.sleep(0.001)
            continue
        frame = snapshot.copy()

        blob, scale, dx, dy = preprocess(frame, input_shape)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: blob})
        infer_ms = (time.perf_counter() - t0) * 1000

        raw_out = outputs[0]
        assert isinstance(raw_out, np.ndarray)

        detections = postprocess(raw_out, frame.shape, scale, dx, dy)

        with result_lock:
            latest_result = (frame, detections, infer_ms)


def render_loop():
    fps = 0.0
    prev = time.perf_counter()
    infer_ms = 0.0
    win_created = False
    detections: list = []

    while running.is_set():
        with result_lock:
            data = latest_result
        with capture_lock:
            raw = latest_capture

        if raw is None:
            time.sleep(0.001)
            continue
        if data is not None:
            _, detections, infer_ms = data

        frame = raw.copy()
        h, w  = frame.shape[:2]

        for x1, y1, x2, y2, conf, cid in detections:
            colour = COLOUR_LUT[min(cid, len(COLOUR_LUT) - 1)]
            label  = f"{class_label(cid)}  {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            (tw, th), bl = cv2.getTextSize(label, FONT, 0.55, 1)
            ty = max(y1 - 4, th + bl + 2)
            cv2.rectangle(frame, (x1, ty - th - bl - 2), (x1 + tw + 6, ty + bl), colour, -1)
            cv2.putText(frame, label, (x1 + 3, ty - 1), FONT, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

        now = time.perf_counter()
        dt = now - prev
        prev = now
        fps = fps * 0.9 + (1.0 / max(dt, 1e-6)) * 0.1

        cv2.putText(frame, f"FPS: {fps:.1f}  |  Infer: {infer_ms:.1f} ms  |  Det: {len(detections)}", (20, 40), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if not win_created:
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_NAME, INIT_WIN_W, INIT_WIN_H)
            win_created = True

        rect = cv2.getWindowImageRect(WIN_NAME)
        win_w = max(1, rect[2])
        win_h = max(1, rect[3])
        display = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(WIN_NAME, display)

        key_wait = max(1, int((RENDER_FRAME_TIME - (time.perf_counter() - now)) * 1000))
        if cv2.waitKey(key_wait) & 0xFF in (27, ord("q"), ord("Q")):
            running.clear()
            break

    cv2.destroyAllWindows()


def main():
    global CAPTURE_SOURCE, CAMERA_INDEX

    parser = argparse.ArgumentParser(description="YOLO ONNX real-time inference")
    parser.add_argument("--source", choices=["camera", "desktop"], default="camera")
    parser.add_argument("--camera-index", type=int, default=None)
    args = parser.parse_args()

    CAPTURE_SOURCE = args.source
    CAMERA_INDEX = (
        args.camera_index if args.camera_index is not None
        else (find_logitech_camera() if CAPTURE_SOURCE == "camera" else 0)
    )

    if not Path(MODEL_PATH).exists():
        return

    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_meta  = session.get_inputs()[0]
    input_name  = input_meta.name
    input_shape: tuple[int, ...] = tuple(
        d if isinstance(d, int) else (1 if i == 0 else 640)
        for i, d in enumerate(input_meta.shape)
    )

    assert len(input_shape) == 4
    assert input_shape[1] == 3

    t_capture   = threading.Thread(target=capture_worker, daemon=True)
    t_inference = threading.Thread(target=inference_worker, args=(session, input_name, input_shape), daemon=True)
    t_capture.start()
    t_inference.start()

    render_loop()

    running.clear()
    t_capture.join(timeout=3)
    t_inference.join(timeout=3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        running.clear()