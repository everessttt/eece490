import re
import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = "/home/everest/EECE490/quantize/runs"

def parse_logs(quant_dir):
    """Parse all numbered subfolders in a quantization directory."""
    bits_list = []
    P_list = []
    R_list = []
    map50_list = []
    map5095_list = []
    inference_list = []

    if not os.path.exists(quant_dir):
        print(f"Directory not found: {quant_dir}")
        return None

    subfolders = sorted([
        d for d in os.listdir(quant_dir)
        if os.path.isdir(os.path.join(quant_dir, d)) and d.isdigit()
    ], key=lambda x: int(x))

    for subfolder in subfolders:
        bits = int(subfolder)
        log_file = os.path.join(quant_dir, subfolder, "val.log")

        if not os.path.exists(log_file):
            print(f"  Skipping bits={bits}: val.log not found")
            continue

        with open(log_file, 'r') as f:
            content = f.read()

        content = re.sub(r'\x1b\[[0-9;]*m', '', content)
        content = re.sub(r'\[K', '', content)

        # Parse mAP/P/R
        match = re.search(
            r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            content
        )

        # Parse speed
        speed_match = re.search(
            r'Speed:.*?([\d.]+)ms preprocess,\s*([\d.]+)ms inference,\s*([\d.]+)ms loss,\s*([\d.]+)ms postprocess',
            content
        )

        if match:
            P, R, map50, map5095 = [float(x) for x in match.groups()]
            bits_list.append(bits)
            P_list.append(P)
            R_list.append(R)
            map50_list.append(map50)
            map5095_list.append(map5095)

            if speed_match:
                inference_ms = float(speed_match.group(2))
                inference_list.append(inference_ms)
                print(f"  bits={bits}: P={P:.3f} R={R:.3f} mAP50={map50:.3f} mAP50-95={map5095:.3f} inference={inference_ms:.1f}ms")
            else:
                inference_list.append(None)
                print(f"  bits={bits}: P={P:.3f} R={R:.3f} mAP50={map50:.3f} mAP50-95={map5095:.3f} inference=N/A")
        else:
            print(f"  bits={bits}: Could not parse val.log")

    if not bits_list:
        return None

    bits_arr    = np.array(bits_list)[::-1]
    P_arr       = np.array(P_list)[::-1]
    R_arr       = np.array(R_list)[::-1]
    map50_arr   = np.array(map50_list)[::-1]
    map5095_arr = np.array(map5095_list)[::-1]
    f1_arr      = 2 * (P_arr * R_arr) / (P_arr + R_arr + 1e-8)
    inference_arr = np.array(inference_list[::-1], dtype=float)

    return {
        'bits': bits_arr,
        'P': P_arr,
        'R': R_arr,
        'map50': map50_arr,
        'map5095': map5095_arr,
        'f1': f1_arr,
        'inference': inference_arr,
        'f1_eff': f1_arr / bits_arr,
        'map50_eff': map50_arr / bits_arr,
        'map5095_eff': map5095_arr / bits_arr,
    }


def make_plot(bits_arr, y_arr, eff_arr, best_bit, best_eff, ylabel, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=10, fontweight='bold')

    axes[0].plot(bits_arr, y_arr, 'o-', color='black', linewidth=2, markersize=4)
    axes[0].axvline(x=best_bit, color='red', linestyle='--', linewidth=1, label=f'Peak at {best_bit}-bit')
    axes[0].set_xlabel("Num Bits", fontsize=10)
    axes[0].set_ylabel(ylabel, fontsize=10)
    axes[0].set_title(f"{ylabel} vs Num Bits", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(bits_arr[::2])
    axes[0].invert_xaxis()
    axes[0].legend()

    axes[1].plot(bits_arr, eff_arr, 'o-', color='black', linewidth=2, markersize=4)
    axes[1].axvline(x=best_bit, color='red', linestyle='--', linewidth=1,
                    label=f'Peak at {best_bit}-bit ({ylabel}/bit={best_eff:.4f})')
    axes[1].set_xlabel("Num Bits", fontsize=10)
    axes[1].set_ylabel(f"{ylabel} / Num Bits", fontsize=10)
    axes[1].set_title(f"Efficiency ({ylabel} per Bit) vs Num Bits", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(bits_arr[::2])
    axes[1].invert_xaxis()
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_inference(bits_arr, inference_arr, out_path, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"YOLO11n {label}: Inference Time vs Num Bits", fontsize=10, fontweight='bold')

    ax.plot(bits_arr, inference_arr, 'o-', color='black', linewidth=2, markersize=4)
    ax.set_xlabel("Num Bits", fontsize=10)
    ax.set_ylabel("Inference Time (ms)", fontsize=10)
    ax.set_title("Inference Time vs Num Bits", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_arr[::2])
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_metrics(data, label, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    bits_arr    = data['bits']
    f1_arr      = data['f1']
    map50_arr   = data['map50']
    map5095_arr = data['map5095']
    f1_eff      = data['f1_eff']
    map50_eff   = data['map50_eff']
    map5095_eff = data['map5095_eff']
    inference_arr = data['inference']

    best_f1_bit      = bits_arr[np.argmax(f1_eff)]
    best_map50_bit   = bits_arr[np.argmax(map50_eff)]
    best_map5095_bit = bits_arr[np.argmax(map5095_eff)]

    make_plot(bits_arr, f1_arr, f1_eff, best_f1_bit,
              f1_eff[np.argmax(f1_eff)],
              "F1 Score", f"YOLO11n {label}: F1 vs Num Bits",
              os.path.join(out_dir, "f1_tradeoff.png"))

    make_plot(bits_arr, map50_arr, map50_eff, best_map50_bit,
              map50_eff[np.argmax(map50_eff)],
              "mAP50", f"YOLO11n {label}: mAP50 vs Num Bits",
              os.path.join(out_dir, "map50_tradeoff.png"))

    make_plot(bits_arr, map5095_arr, map5095_eff, best_map5095_bit,
              map5095_eff[np.argmax(map5095_eff)],
              "mAP50-95", f"YOLO11n {label}: mAP50-95 vs Num Bits",
              os.path.join(out_dir, "map5095_tradeoff.png"))

    # Inference time plot (only if data available)
    if not np.all(np.isnan(inference_arr)):
        plot_inference(bits_arr, inference_arr,
                       os.path.join(out_dir, "inference_tradeoff.png"), label)

    # Combined overview: mAP50, F1, inference on one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"YOLO11n {label}: Overview", fontsize=10, fontweight='bold')

    axes[0].plot(bits_arr, map50_arr, 'o-', color='black', linewidth=2, markersize=4)
    axes[0].axvline(x=best_map50_bit, color='red', linestyle='--', linewidth=1, label=f'Peak eff @ {best_map50_bit}-bit')
    axes[0].set_xlabel("Num Bits", fontsize=10)
    axes[0].set_ylabel("mAP50", fontsize=10)
    axes[0].set_title("mAP50 vs Num Bits", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(bits_arr[::2])
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=8)

    axes[1].plot(bits_arr, f1_arr, 'o-', color='black', linewidth=2, markersize=4)
    axes[1].axvline(x=best_f1_bit, color='red', linestyle='--', linewidth=1, label=f'Peak eff @ {best_f1_bit}-bit')
    axes[1].set_xlabel("Num Bits", fontsize=10)
    axes[1].set_ylabel("F1 Score", fontsize=10)
    axes[1].set_title("F1 vs Num Bits", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(bits_arr[::2])
    axes[1].invert_xaxis()
    axes[1].legend(fontsize=8)

    if not np.all(np.isnan(inference_arr)):
        axes[2].plot(bits_arr, inference_arr, 'o-', color='black', linewidth=2, markersize=4)
        axes[2].set_xlabel("Num Bits", fontsize=10)
        axes[2].set_ylabel("Inference Time (ms)", fontsize=10)
        axes[2].set_title("Inference Time vs Num Bits", fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(bits_arr[::2])
        axes[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overview.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: overview.png")


# --- Run ---
for quant_type in ["symmetric", "asymmetric"]:
    quant_dir = os.path.join(base_dir, quant_type)
    print(f"\nParsing {quant_type}...")
    data = parse_logs(quant_dir)
    if data:
        plot_metrics(data, label=quant_type.capitalize(), out_dir=quant_dir)
    else:
        print(f"  No data found for {quant_type}")

print("\nDone.")