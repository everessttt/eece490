"""
plot_results.py

Parses all val.log files under RUNS_ROOT and generates:
  - mAP50 vs bit-width
  - mAP50-95 vs bit-width
  - F1 vs bit-width (estimated as 2*P*R / (P+R))
  - Inference time vs bit-width
  - mAP50 / bit-width vs bit-width
  - F1 / bit-width vs bit-width
  - Delta plots (bit-by-bit difference) for mAP50, mAP50-95, F1
  - Quantization error (%) vs bit-width (computed from weights)

Usage:
    python plot.py
"""

import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config
import torch
from ultralytics import YOLO
from quantization_helpers import quantization_methods

ROOT = Path(config.ROOT)
RUNS_ROOT = ROOT / "runs" / "quantized_models"
MODEL_ROOT = ROOT / "models"
OUTPUT_DIR = ROOT / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Parsing
# -------------------------------------------------------------------

def parse_log(log_path: Path) -> dict | None:
    text = log_path.read_text(errors="ignore")

    metrics = re.search(
        r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text
    )
    speed = re.search(r"([\d.]+)ms inference", text)

    if not metrics:
        return None

    precision = float(metrics.group(1))
    recall = float(metrics.group(2))
    map50 = float(metrics.group(3))
    map5095 = float(metrics.group(4))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    inference_ms = float(speed.group(1)) if speed else None

    return {
        "P": precision,
        "R": recall,
        "F1": f1,
        "mAP50": map50,
        "mAP50-95": map5095,
        "inference_ms": inference_ms,
    }


def collect_results() -> dict:
    results = {}

    for log_path in RUNS_ROOT.rglob("val.log"):
        parts = log_path.parts
        try:
            qm_idx = parts.index("quantized_models") + 1
            method = parts[qm_idx]
            stem = parts[qm_idx + 3]
        except (ValueError, IndexError):
            continue

        bits_match = re.search(r"_(\d+)bit$", stem)
        if not bits_match:
            continue
        bits = int(bits_match.group(1))
        model_name = stem[:bits_match.start()]

        metrics = parse_log(log_path)
        if metrics is None:
            continue

        metrics["mAP50_per_bit"] = metrics["mAP50"] / bits
        metrics["F1_per_bit"] = metrics["F1"] / bits
        metrics["log_path"] = log_path

        results.setdefault(method, {}).setdefault(model_name, {})[bits] = metrics

    return results


# -------------------------------------------------------------------
# Quantization error
# -------------------------------------------------------------------

def compute_quant_error(pt_path: Path, quant_fn, bits: int) -> float:
    yolo = YOLO(str(pt_path))
    model: torch.nn.Module = yolo.model  # type: ignore
    errors = []
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dtype != torch.float32 or param.ndim < 2:
                continue
            original = param.data.clone()
            quantized = quant_fn(original, num_bits=bits)
            abs_orig = original.abs()
            nonzero = abs_orig > 0
            if nonzero.any():
                pct_err = (
                    (original - quantized).abs()[nonzero] / abs_orig[nonzero]
                ).mean().item() * 100
                errors.append(pct_err)
    return float(np.mean(errors)) if errors else 0.0


def collect_quant_errors() -> dict:
    errors = {}
    pt_files = list(MODEL_ROOT.rglob("*.pt"))

    for pt_path in pt_files:
        model_name = pt_path.stem
        print(f"Computing quantization errors for {model_name}...")
        for quant_fn in quantization_methods:
            method = quant_fn.__name__
            max_bits = 8 if method == "PoTPTQ" else 32
            for bits in range(2, max_bits + 1):
                err = compute_quant_error(pt_path, quant_fn, bits)
                errors.setdefault(method, {}).setdefault(model_name, {})[bits] = err
                print(f"  {method} {bits}bit: {err:.4f}%", end="\r", flush=True)
        print()

    return errors


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def _setup_ax(ax, xlabel: str, ylabel: str, title: str,
              log_scale: bool = False, bottom: float | None = 0):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(2, 33, 2))
    ax.invert_xaxis()
    ax.axhline(y=0, color="black", linewidth=1.5, linestyle="-")
    if bottom is not None:
        ax.set_ylim(bottom=bottom)
    if log_scale:
        ax.set_yscale("log")


# -------------------------------------------------------------------
# Plot functions
# -------------------------------------------------------------------

def plot_metric(results: dict, metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = sorted(results.keys())
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(methods)))

    for color, method in zip(colors, methods):
        for model_name, bit_data in results[method].items():
            bits_sorted = sorted(bit_data.keys())
            values = [bit_data[b].get(metric) for b in bits_sorted]
            valid = [(b, v) for b, v in zip(bits_sorted, values) if v is not None]
            if not valid:
                continue
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker="o", markersize=3,
                    label=f"{method} / {model_name}", color=color)

            log_path = bit_data[bits_sorted[0]]["log_path"]
            data_dir = log_path.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                data_dir / f"{metric}.csv",
                np.array(list(zip(xs, ys))),
                delimiter=",",
                header="bits,value",
                comments=""
            )

    _setup_ax(ax, "Bit-width", ylabel, f"{ylabel} vs Bit-width")

    out = OUTPUT_DIR / f"{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_metric_delta(results: dict, metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = sorted(results.keys())
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(methods)))

    for color, method in zip(colors, methods):
        for model_name, bit_data in results[method].items():
            bits_sorted = sorted(bit_data.keys(), reverse=True)
            values = [bit_data[b].get(metric) for b in bits_sorted]
            valid = [(b, v) for b, v in zip(bits_sorted, values) if v is not None]
            if len(valid) < 2:
                continue

            xs = [valid[i + 1][0] for i in range(len(valid) - 1)]
            ys = [valid[i][1] - valid[i + 1][1] for i in range(len(valid) - 1)]

            ax.plot(xs, ys, marker="o", markersize=3,
                    label=f"{method} / {model_name}", color=color)

            log_path = bit_data[bits_sorted[-1]]["log_path"]
            data_dir = log_path.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                data_dir / f"{metric}_delta.csv",
                np.array(list(zip(xs, ys))),
                delimiter=",",
                header="bits,delta",
                comments=""
            )

    _setup_ax(ax, "Bit-width (lower of pair)", f"Δ {ylabel}",
              f"Δ {ylabel} per Bit Step", bottom=None)

    out = OUTPUT_DIR / f"{metric}_delta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_quant_error(errors: dict):
    if not errors:
        print("No quantization error data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    methods = sorted(errors.keys())
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(methods)))
    has_data = False

    for color, method in zip(colors, methods):
        for model_name, bit_data in errors[method].items():
            bits_sorted = sorted(bit_data.keys())
            values = [bit_data[b] for b in bits_sorted]
            if not any(v > 0 for v in values):
                continue
            has_data = True
            ax.plot(bits_sorted, values, marker="o", markersize=3,
                    label=f"{method} / {model_name}", color=color)

            data_dir = RUNS_ROOT / method / "det" / model_name / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                data_dir / "quantization_error.csv",
                np.array(list(zip(bits_sorted, values))),
                delimiter=",",
                header="bits,pct_error",
                comments=""
            )

    if not has_data:
        print("No positive quantization error values to plot.")
        plt.close(fig)
        return

    _setup_ax(ax, "Bit-width", "Mean Absolute Quantization Error (%)",
              "Quantization Error vs Bit-width", log_scale=True)

    out = OUTPUT_DIR / "quantization_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

METRICS = {
    "mAP50":         "mAP@50",
    "mAP50-95":      "mAP@50-95",
    "F1":            "F1 Score",
    "inference_ms":  "Inference Time (ms)",
    "mAP50_per_bit": "mAP@50 / Bit-width",
    "F1_per_bit":    "F1 / Bit-width",
}

DELTA_METRICS = {
    "mAP50":    "mAP@50",
    "mAP50-95": "mAP@50-95",
    "F1":       "F1 Score",
}

if __name__ == "__main__":
    print("Collecting validation results from logs...")
    results = collect_results()

    if not results:
        print("No valid val.log files found. Check RUNS_ROOT path.")
        sys.exit(1)

    for metric, ylabel in METRICS.items():
        plot_metric(results, metric, ylabel)

    for metric, ylabel in DELTA_METRICS.items():
        plot_metric_delta(results, metric, ylabel)

    print("\nComputing quantization errors from .pt weights...")
    errors = collect_quant_errors()
    plot_quant_error(errors)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")