import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import config
import torch
from ultralytics import YOLO
from quantization_helpers import quantization_methods

ROOT       = Path(config.ROOT)
RUNS_ROOT  = ROOT / "runs" / "quantized_models"
MODEL_ROOT = ROOT / "models"
OUTPUT_DIR = ROOT / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PARAMS = {
    "yolo11n":      2_624_080,
    "yolo11s":      9_458_752,
    "yolo11m":     20_114_688,
    "yolo11l":     25_372_160,
    "yolo11x":     56_966_176,
    "yolo11n-seg":  2_876_848,
    "yolo11s-seg": 10_113_248,
    "yolo11m-seg": 22_420_896,
    "yolo11l-seg": 27_678_368,
    "yolo11x-seg": 62_142_656,
    "yolo11n-pose": 2_874_462,
    "yolo11s-pose": 9_918_238,
    "yolo11m-pose":20_912_364,
    "yolo11l-pose":26_169_836,
    "yolo11x-pose":58_798_636,
}

TASKS = ("det", "seg", "pose")
SIZES = ("n", "s", "m", "l", "x")
BITS  = tuple(range(2, 33))

SIZE_COLORS = {
    "n": "#FF0000",
    "s": "#FF7F00",
    "m": "#00FF00",
    "l": "#0000FF",
    "x": "#9400D3"
}

TASK_COLORS = {
    "det":  "#FF0000",
    "seg":  "#00FF00",
    "pose": "#0000FF"
}

QUANT_METHOD_COLORS = {
    "uniform_affine":    "#FF0000",
    "uniform_symmetric": "#00FF00",
    "power_of_two":      "#0000FF"
}

METRICS = {
    "mAP50":            "mAP@50",
    "mAP50-95":         "mAP@50-95",
    "F1":               "F1 Score",
    "inference_ms":     "Inference Time (ms)",
    "mAP50_per_bit":    "mAP@50 per Num Bits",
    "F1_per_bit":       "F1 per Num Bits"
}
DELTA_METRICS   = {"mAP50": "mAP@50", "mAP50-95": "mAP@50-95", "F1": "F1 Score"}
FAMILY_METRICS  = {"mAP50": "mAP@50", "mAP50-95": "mAP@50-95", "F1": "F1 Score"}

#---helpers---

def get_task(model_name: str) -> str:
    if "seg"  in model_name: return "seg"
    if "pose" in model_name: return "pose"
    return "det"

def get_size(model_name: str) -> str:
    return model_name.replace("yolo11", "").split("-")[0]

def get_params(model_name: str) -> Optional[int]:
    return MODEL_PARAMS.get(model_name)

def _bit_idx(bits: int) -> int:
    return bits - 2

@dataclass
class Series:
    label:  str
    color:  str
    x_data: np.ndarray
    y_data: np.ndarray

@dataclass
class PlotData:
    x_label: str
    y_label: str
    title: str
    save_dir: Path
    x_col_name: str = "x"
    y_col_name: str = "y"

    _series_keys: list = field(default_factory=list)
    _series_colors: dict = field(default_factory=dict)
    _series_labels: dict = field(default_factory=dict)
    _x_raw: np.ndarray = field(default_factory=lambda: np.full((len(BITS), 0), np.nan))
    _y_raw: np.ndarray = field(default_factory=lambda: np.full((len(BITS), 0), np.nan))

    series: list = field(default_factory=list)

    @staticmethod
    def for_sizes(x_label: str, y_label: str, title: str, save_dir: Path, x_col: str = "x", y_col: str = "y") -> "PlotData":
        n = len(SIZES)
        pd = PlotData(x_label, y_label, title, save_dir, x_col, y_col, list(SIZES), SIZE_COLORS, {sz: f"yolo11{sz}" for sz in SIZES}, np.full((len(BITS), n), np.nan), np.full((len(BITS), n), np.nan))
        return pd

    @staticmethod
    def for_tasks(x_label: str, y_label: str, title: str, save_dir: Path, x_col: str = "x", y_col: str = "y") -> "PlotData":
        n = len(TASKS)
        pd = PlotData(x_label, y_label, title, save_dir, x_col, y_col, list(TASKS), TASK_COLORS, {t: t for t in TASKS}, np.full((len(BITS), n), np.nan), np.full((len(BITS), n), np.nan))
        return pd
    
    @staticmethod
    def for_methods(methods: list, x_label: str, y_label: str, title: str, save_dir: Path, x_col: str = "x", y_col: str = "y") -> "PlotData":
        n = len(methods)
        pd = PlotData(x_label, y_label, title, save_dir, x_col, y_col, list(methods), QUANT_METHOD_COLORS, {m: m for m in methods}, np.full((len(BITS), n), np.nan), np.full((len(BITS), n), np.nan))
        return pd

    def set(self, key: str, bit_idx: int, x: float, y: float):
        col = self._series_keys.index(key)
        self._x_raw[bit_idx, col] = x
        self._y_raw[bit_idx, col] = y

    def build(self) -> "PlotData":
        self.series = []
        for ci, key in enumerate(self._series_keys):
            mask = ~np.isnan(self._y_raw[:, ci])
            if not mask.any():
                continue
            self.series.append(Series(label  = self._series_labels[key], color  = self._series_colors[key], x_data = self._x_raw[mask, ci], y_data = self._y_raw[mask, ci]))
        return self

    def has_data(self) -> bool:
        return bool(self.series)

    def save_csv(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for s in self.series:
            np.savetxt(self.save_dir / f"{s.label}.csv", np.column_stack([s.x_data, s.y_data]), delimiter=",", header=f"{self.x_col_name},{self.y_col_name}", comments="")

def make_plot(
    data: PlotData,
    out_path: Path,
    *,
    log_x: bool = False,
    log_y: bool = False,
    invert_x: bool = False,
    y_bottom: Optional[float] = None,
    annotate: bool = False,
    connect_dots: bool = True,
    figsize: tuple = (10, 6),
):
    if not data.has_data():
        return

    fig, ax = plt.subplots(figsize=figsize)

    for s in data.series:
        if connect_dots:
            ax.plot(s.x_data, s.y_data, marker="o", markersize=4, color=s.color, linewidth=1.5, label=s.label, alpha=0.85)
        else:
            ax.scatter(s.x_data, s.y_data, s=20, color=s.color, alpha=0.7, label=s.label)
        if annotate:
            ax.annotate(s.label, (s.x_data[-1], s.y_data[-1]), textcoords="offset points", xytext=(4, 4), fontsize=6, color=s.color, alpha=0.9)

    handles = [Line2D([0], [0], color=s.color, marker="o", markersize=5, linewidth=1.5, label=s.label) for s in data.series]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel(data.x_label)
    ax.set_ylabel(data.y_label)
    ax.set_title(data.title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=1.0, linestyle="-")

    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    if invert_x: ax.invert_xaxis()
    if y_bottom is not None: ax.set_ylim(bottom=y_bottom)
    if not log_x: ax.set_xticks(range(2, 33, 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    data.save_csv()

#---build---

def build_metric_by_task(
    method_data: dict, task: str, metric: str, ylabel: str,
    method: str, save_root: Path,
) -> PlotData:
    pd = PlotData.for_sizes(
        "Num Bits", ylabel,
        f"{ylabel} vs Num Bits <- {method} ({task})",
        save_root / metric / method / task, "bits", metric,
    )
    for model_name, bit_data in method_data.items():
        if get_task(model_name) != task:
            continue
        sz = get_size(model_name)
        for bits, data in bit_data.items():
            v = data.get(metric)
            if v is not None and 2 <= bits <= 32:
                pd.set(sz, _bit_idx(bits), bits, v)
    return pd.build()

def build_metric_compare_tasks(
    method_data: dict, size: str, metric: str, ylabel: str,
    method: str, save_root: Path,
) -> PlotData:
    pd = PlotData.for_tasks("Num Bits", ylabel, f"{ylabel} vs Num Bits ({method}) (yolo11{size}) (all tasks)", save_root / metric / method / "compare_tasks" / size, "bits", metric)
    for model_name, bit_data in method_data.items():
        if get_size(model_name) != size:
            continue
        task = get_task(model_name)
        for bits, data in bit_data.items():
            v = data.get(metric)
            if v is not None and 2 <= bits <= 32:
                pd.set(task, _bit_idx(bits), bits, v)
    return pd.build()

def _delta_pairs(bit_data: dict, metric: str) -> list:
    vals = sorted((b, d[metric]) for b, d in bit_data.items() if d.get(metric) is not None)
    return [(vals[i][0], vals[i + 1][1] - vals[i][1]) for i in range(len(vals) - 1)]

def build_delta_by_task(
    method_data: dict, task: str, metric: str, ylabel: str,
    method: str, save_root: Path,
) -> PlotData:
    pd = PlotData.for_sizes("Num Bits (lower of pair)", f"Δ {ylabel}", f"Δ {ylabel} per Num Bits ({method}) ({task})", save_root / f"{metric}_delta" / method / task, "bits", f"delta_{metric}")
    for model_name, bit_data in method_data.items():
        if get_task(model_name) != task:
            continue
        sz = get_size(model_name)
        for lo_bits, delta in _delta_pairs(bit_data, metric):
            if 2 <= lo_bits <= 32:
                pd.set(sz, _bit_idx(lo_bits), lo_bits, delta)
    return pd.build()

def build_delta_compare_tasks(
    method_data: dict, size: str, metric: str, ylabel: str,
    method: str, save_root: Path,
) -> PlotData:
    pd = PlotData.for_tasks("Num Bits (lower of pair)", f"Δ {ylabel}", f"Δ {ylabel} per Num Bits ({method}) (yolo11{size}) (all tasks)", save_root / f"{metric}_delta" / method / "compare_tasks" / size, "bits", f"delta_{metric}")
    for model_name, bit_data in method_data.items():
        if get_size(model_name) != size:
            continue
        task = get_task(model_name)
        for lo_bits, delta in _delta_pairs(bit_data, metric):
            if 2 <= lo_bits <= 32:
                pd.set(task, _bit_idx(lo_bits), lo_bits, delta)
    return pd.build()

def build_delta_compare_methods(
    results: dict, model_name: str, metric: str, ylabel: str,
    save_root: Path,
) -> PlotData:
    methods = sorted(results.keys())
    pd = PlotData.for_methods(methods, "Num Bits (lower of pair)", f"Δ {ylabel}", f"Δ {ylabel} per Num Bits ({model_name}) (all methods)", save_root / f"{metric}_delta" / model_name, "bits", f"delta_{metric}")
    for method in methods:
        bit_data = results[method].get(model_name, {})
        for lo_bits, delta in _delta_pairs(bit_data, metric):
            if 2 <= lo_bits <= 32:
                pd.set(method, _bit_idx(lo_bits), lo_bits, delta)
    return pd.build()

def build_family_by_task(
    method_data: dict, task: str, metric: str, ylabel: str,
    method: str, save_root: Path,
) -> PlotData:
    pd = PlotData.for_sizes("Parameters × Num Bits (Bit Efficiency)", ylabel, f"{ylabel} vs Bit Efficiency ({method}) ({task})", save_root / f"family_{metric}" / method / task, "effective_bits", metric,)
    for model_name, bit_data in method_data.items():
        if get_task(model_name) != task:
            continue
        params = get_params(model_name)
        if params is None:
            continue
        sz = get_size(model_name)
        for bits, data in bit_data.items():
            v = data.get(metric)
            if v is not None and 2 <= bits <= 32:
                pd.set(sz, _bit_idx(bits), params * bits, v)
    return pd.build()

def build_family_compare_tasks(
    method_data: dict, size: str, metric: str, ylabel: str,
    method: str, save_root: Path,
) -> PlotData:
    pd = PlotData.for_tasks("Parameters × Num Bits (effective bit cost)", ylabel, f"{ylabel} vs Effective Bit Cost ({method}) (yolo11{size}) (all tasks)", save_root / f"family_{metric}" / method / "compare_tasks" / size, "effective_bits", metric)
    for model_name, bit_data in method_data.items():
        if get_size(model_name) != size:
            continue
        params = get_params(model_name)
        if params is None:
            continue
        task = get_task(model_name)
        for bits, data in bit_data.items():
            v = data.get(metric)
            if v is not None and 2 <= bits <= 32:
                pd.set(task, _bit_idx(bits), params * bits, v)
    return pd.build()

def build_metric_compare_methods(
    results: dict, model_name: str, metric: str, ylabel: str,
    save_root: Path,
) -> PlotData:
    methods = sorted(results.keys())
    pd = PlotData.for_methods(methods, "Num Bits", ylabel, f"{ylabel} vs Num Bits ({model_name}) (all methods)", save_root / metric / model_name, "bits", metric)
    for method in methods:
        bit_data = results[method].get(model_name, {})
        for bits, data in bit_data.items():
            v = data.get(metric)
            if v is not None and 2 <= bits <= 32:
                pd.set(method, _bit_idx(bits), bits, v)
    return pd.build()

def build_family_compare_methods(
    results: dict, model_name: str, metric: str, ylabel: str,
    save_root: Path,
) -> PlotData:
    params = get_params(model_name)
    if params is None:
        return PlotData("", "", "", save_root).build()
    methods = sorted(results.keys())
    pd = PlotData.for_methods(methods, "Parameters × Num Bits (Bit Efficiency)", ylabel, f"{ylabel} vs Effective Bit Cost ({model_name}) (all methods)", save_root / f"family_{metric}" / model_name, "effective_bits", metric)
    for method in methods:
        bit_data = results[method].get(model_name, {})
        for bits, data in bit_data.items():
            v = data.get(metric)
            if v is not None and 2 <= bits <= 32:
                pd.set(method, _bit_idx(bits), params * bits, v)
    return pd.build()

def build_quant_error_by_method(
    method_data: dict, method: str, save_root: Path,
) -> PlotData:
    pd = PlotData.for_sizes("Num Bits", "Mean Absolute Quantization Error (%)", f"Quantization Error vs Num Bits ({method})", save_root / "quantization_error" / method, "bits", "percent_error")
    for model_name, bit_data in method_data.items():
        sz = get_size(model_name)
        for bits, err in bit_data.items():
            if err > 0 and 2 <= bits <= 32:
                pd.set(sz, _bit_idx(bits), bits, err)
    return pd.build()

#---parse data---

def parse_log(log_path: Path) -> Optional[dict]:
    text = log_path.read_text(errors="ignore")
    m = re.search(r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    s = re.search(r"([\d.]+)ms inference", text)
    if not m:
        return None
    precision, recall = float(m.group(1)), float(m.group(2))
    map50, map5095    = float(m.group(3)), float(m.group(4))
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {
        "P": precision, "R": recall, "F1": f1,
        "mAP50": map50, "mAP50-95": map5095,
        "inference_ms": float(s.group(1)) if s else None,
    }

def collect_results() -> dict:
    results = {}
    for log_path in RUNS_ROOT.rglob("val.log"):
        parts = log_path.parts
        try:
            qm_idx = parts.index("quantized_models") + 1
            method = parts[qm_idx]
            stem   = parts[qm_idx + 3]
        except (ValueError, IndexError):
            continue

        bits_match = re.search(r"_(\d+)bit$", stem)
        if bits_match:
            bits, model_name = int(bits_match.group(1)), stem[: bits_match.start()]
        else:
            int_match = re.search(r"_(int(\d+))$", stem)
            if not int_match:
                continue
            bits, model_name = int(int_match.group(2)), stem[: int_match.start()]

        metrics = parse_log(log_path)
        if metrics is None:
            continue
        metrics["mAP50_per_bit"] = metrics["mAP50"] / bits
        metrics["F1_per_bit"]    = metrics["F1"] / bits
        metrics["log_path"]      = log_path
        results.setdefault(method, {}).setdefault(model_name, {})[bits] = metrics

    return results

def compute_quant_error(pt_path: Path, quant_fn: Callable, bits: int) -> float:
    model: torch.nn.Module = YOLO(str(pt_path)).model  # type: ignore
    errors = []
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dtype != torch.float32 or param.ndim < 2:
                continue
            original  = param.data.clone()
            quantized = quant_fn(original, num_bits=bits)
            abs_orig  = original.abs()
            nonzero   = abs_orig > 0
            if nonzero.any():
                errors.append(((original - quantized).abs()[nonzero] / abs_orig[nonzero]).mean().item() * 100)
    return float(np.mean(errors)) if errors else 0.0

def collect_quant_errors() -> dict:
    errors = {}
    for pt_path in MODEL_ROOT.rglob("*.pt"):
        model_name = pt_path.stem
        print(f"Computing quantization errors for {model_name}...")
        for quant_fn in quantization_methods:
            method = quant_fn.__name__
            max_bits = 8 if method == "PoTPTQ" else 32
            for bits in range(2, max_bits + 1):
                err = compute_quant_error(pt_path, quant_fn, bits)
                errors.setdefault(method, {}).setdefault(model_name, {})[bits] = err
    return errors

#---plot---

PLOT_ROOT   = OUTPUT_DIR / "model"
FAMILY_ROOT = OUTPUT_DIR / "model_family"
DATA_ROOT   = OUTPUT_DIR / "data"

def _metric_plot_kwargs() -> dict:
    return dict(invert_x=True, y_bottom=0)

def _delta_plot_kwargs() -> dict:
    return dict(invert_x=True)

def _family_plot_kwargs() -> dict:
    return dict(log_x=True, invert_x=True, y_bottom=0, annotate=True, connect_dots=True)

def _quant_error_kwargs() -> dict:
    return dict(log_y=True, invert_x=True)

def plot_metrics(results: dict):
    for method, method_data in results.items():
        for metric, ylabel in METRICS.items():
            for task in TASKS:
                pd = build_metric_by_task(method_data, task, metric, ylabel, method, DATA_ROOT)
                make_plot(pd, PLOT_ROOT / metric / method / f"{task}.png", **_metric_plot_kwargs())
            for size in SIZES:
                pd = build_metric_compare_tasks(method_data, size, metric, ylabel, method, DATA_ROOT)
                make_plot(pd, PLOT_ROOT / metric / method / "compare_tasks" / f"{size}.png", **_metric_plot_kwargs())

def plot_deltas(results: dict):
    for method, method_data in results.items():
        for metric, ylabel in DELTA_METRICS.items():
            for task in TASKS:
                pd = build_delta_by_task(method_data, task, metric, ylabel, method, DATA_ROOT)
                make_plot(pd, PLOT_ROOT / f"{metric}_delta" / method / f"{task}.png", **_delta_plot_kwargs())
            for size in SIZES:
                pd = build_delta_compare_tasks(method_data, size, metric, ylabel, method, DATA_ROOT)
                make_plot(pd, PLOT_ROOT / f"{metric}_delta" / method / "compare_tasks" / f"{size}.png", **_delta_plot_kwargs())

def plot_family(results: dict):
    for method, method_data in results.items():
        for metric, ylabel in FAMILY_METRICS.items():
            for task in TASKS:
                pd = build_family_by_task(method_data, task, metric, ylabel, method, DATA_ROOT)
                make_plot(pd, FAMILY_ROOT / metric / method / f"{task}.png", **_family_plot_kwargs())
            for size in SIZES:
                pd = build_family_compare_tasks(method_data, size, metric, ylabel, method, DATA_ROOT)
                make_plot(pd, FAMILY_ROOT / metric / method / "compare_tasks" / f"{size}.png", **_family_plot_kwargs())

def plot_compare_methods(results: dict, model_name: str = "yolo11n"):
    for metric, ylabel in METRICS.items():
        pd = build_metric_compare_methods(results, model_name, metric, ylabel, DATA_ROOT)
        make_plot(pd, PLOT_ROOT / metric / f"{model_name}.png", **_metric_plot_kwargs())
 
def plot_family_compare_methods(results: dict, model_name: str = "yolo11n"):
    for metric, ylabel in FAMILY_METRICS.items():
        pd = build_family_compare_methods(results, model_name, metric, ylabel, DATA_ROOT)
        make_plot(pd, FAMILY_ROOT / metric / f"{model_name}.png", **_family_plot_kwargs())

def plot_delta_compare_methods(results: dict, model_name: str = "yolo11n"):
    for metric, ylabel in DELTA_METRICS.items():
        pd = build_delta_compare_methods(results, model_name, metric, ylabel, DATA_ROOT)
        make_plot(pd, PLOT_ROOT / f"{metric}_delta" / f"{model_name}.png", **_delta_plot_kwargs())

def plot_quant_errors(errors: dict):
    if not errors:
        return
    for method, method_data in errors.items():
        pd = build_quant_error_by_method(method_data, method, DATA_ROOT)
        make_plot(pd, PLOT_ROOT / "quantization_error" / f"{method}.png", **_quant_error_kwargs())

#---main---

if __name__ == "__main__":
    print("Creating plots from validation logs...")
    results = collect_results()

    if not results:
        print("No val.log files found")
        sys.exit(1)

    plot_metrics(results)
    plot_deltas(results)
    plot_family(results)
    plot_compare_methods(results)
    plot_delta_compare_methods(results)
    plot_family_compare_methods(results)

    plot_quant_errors(collect_quant_errors())

    print(f"\nAll plots saved to: {OUTPUT_DIR}")