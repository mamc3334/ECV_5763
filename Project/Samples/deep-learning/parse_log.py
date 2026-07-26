import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import csv

CAPTURE       = 0
PREDICT_START = 1
PREDICT_END   = 2
WB_START      = 3
WB_END        = 4

def _stats(arr, label, unit, f):
    """Print min/max/avg/std for an array."""
    if len(arr) == 0:
        print(f"  {label}: No data available.", file=f)
        return
    print(f"  Min : {np.min(arr):.4f} {unit}", file=f)
    print(f"  Max : {np.max(arr):.4f} {unit}", file=f)
    print(f"  Avg : {np.mean(arr):.4f} {unit}", file=f)
    print(f"  Std : {np.std(arr):.4f} {unit}", file=f)

def parse_log_and_plot(file_path, dir_path):
    frames = {}

    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            t = int(row["time_ns"]) / 1e6       # convert to ms
            frame = int(row["frame"])
            event = int(row["event"])

            if frame not in frames:
                frames[frame] = {}

            frames[frame][event] = t

    frame_sorted = sorted(frames.keys())
    acq_start_times = []
    pro_start_times = []
    pro_done_times = []
    wb_start_times = []
    wb_end_times = []

    for f in frame_sorted:
        ev = frames[f]

        if CAPTURE in ev:
            acq_start_times.append(ev[CAPTURE])

        if PREDICT_START in ev:
            pro_start_times.append(ev[PREDICT_START])

        if PREDICT_END in ev:
            pro_done_times.append(ev[PREDICT_END])

        if WB_START in ev:
            wb_start_times.append(ev[WB_START])

        if WB_END in ev:
            wb_end_times.append(ev[WB_END])

    n_out = len(wb_end_times)

    if n_out == 0:
        print("No log data matched expected patterns. Check file format.")
        return
    
    # Input
    acq_times = np.array(acq_start_times)
    if len(acq_times) > 1:
        acq_inter = np.diff(acq_times)
        acq_inter[acq_inter <= 0] = 1e-3
        input_fps = 1000.0 / acq_inter
    else:
        input_fps = np.array([])

    # Processing
    pro_durations = np.subtract(pro_done_times, pro_start_times)

    # WB
    wb_durations = np.subtract(wb_end_times, wb_start_times)
    wb_times = np.array(wb_end_times)
    if len(wb_times) > 1:
        inter_ms = np.diff(wb_times)
        inter_ms[inter_ms <= 0] = 1e-3
        inst_fps = 1000.0 / inter_ms
        avg_fps = (np.arange(1, len(wb_times)) * 1000.0) / (wb_times[1:] - wb_times[0])
        overall_fps = (1000.0 * (len(wb_times))) / (wb_times[-1] - wb_times[0])
    else:
        inst_fps, avg_fps, overall_fps = np.array([0]), np.array([0]), 0.0

    # ── Terminal Summary ─────────────────────────────────────────────────────
    with open(dir_path / "summary.txt", "w") as f:
        print(f"\nParsed {n_out}  frames", file=f)
        print("=" * 52, file=f)
        print("  PROCESSING EXECUTION TIME per frame (ms)", file=f)
        print("=" * 52, file=f)
        _stats(pro_durations, "Processing Exec Time", "ms", f)
        print("=" * 52, file=f)
        print("  WRITE-BACK EXECUTION TIME per frame (ms)", file=f)
        print("=" * 52, file=f)
        _stats(wb_durations, "Write-back Exec Time", "ms", f)
        print("=" * 52, file=f)
        print("  INPUT CAPTURE RATE (FPS)", file=f)
        print("=" * 52, file=f)
        _stats(input_fps, "Input Rate", "FPS", f)
        print("=" * 52, file=f)
        print("  OUTPUT FPS", file=f)
        print("=" * 52, file=f)
        _stats(inst_fps, "Inst FPS", "FPS", f)
        print(f"  Overall Avg : {overall_fps:.4f} FPS", file=f)
        print("=" * 52, file=f)

    tag = Path(file_path).stem

    # Input Capture Rate vs WB Output Rate
    fig1, (ax1,ax2) = plt.subplots(2, 1, figsize=(9, 7))
    if len(input_fps) > 0:
        ax1.plot(range(1, len(wb_end_times)), input_fps, label='Input Rate (FPS)', color='tab:green', linewidth=1.5)
        avg = np.mean(input_fps[:len(wb_end_times)])
        ax1.axhline(avg, color='black', linestyle='--', linewidth=1.2, label=f'Avg Input FPS ({avg:.2f})')
        
    ax1.set_ylim(0)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Rate (FPS)')
    ax1.set_title(f'Input Frame Rate - {tag}')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    if len(wb_times) > 1:
        ax2.plot(range(1, len(wb_end_times)), inst_fps, label='Inst WB FPS', color='tab:blue', linewidth=1.5)
        ax2.plot(range(1, len(wb_end_times)), avg_fps, label='Running Avg WB FPS', color='tab:red', linewidth=1.5)
        ax2.axhline(overall_fps, color='black', linestyle='--', linewidth=1.2, label=f'Overall Avg WB FPS ({overall_fps:.2f})')

    ax2.set_ylim(0)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Rate (FPS)')
    ax2.set_title(f'Output Frame Rate - {tag}')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    fig1.tight_layout()
    p1 = dir_path / "performance.png"
    fig1.savefig(p1, dpi=300)
    print(f"Performance plot saved to '{p1}'")
    plt.close(fig1)

    # Exec times - PRO and WB
    fig2, (ax3,ax4) = plt.subplots(2, 1, figsize=(9,7))
    if len(pro_durations) > 0:
        ax3.plot(range(0, len(pro_durations)), pro_durations, label='Processing C', color='tab:orange', linewidth=1.5)
        avg_dur = np.mean(pro_durations)
        ax3.axhline(avg_dur, color='orange', linestyle='--', linewidth=1.2, label=f'Avg Processing C ({avg_dur:.2f} ms)')

    ax3.set_ylim(0)
    ax3.set_xlabel('Frame #')
    ax3.set_ylabel('Execution time (ms)')
    ax3.set_title(f'Frame Processing time - {tag}')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()

    if len(wb_durations) > 0:
        ax4.plot(range(0, len(wb_end_times)), wb_durations, label='WB C', color='tab:blue', linewidth=1.5)
        avg_dur = np.mean(wb_durations)
        ax4.axhline(avg_dur, color='blue', linestyle='--', linewidth=1.2, label=f'Avg WB C ({avg_dur:.2f} ms)')

    ax4.set_ylim(0)
    ax4.set_xlabel('Frame #')
    ax4.set_ylabel('Execution time (ms)')
    ax4.set_title(f'Frame Write-Back time - {tag}')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend()

    fig2.tight_layout()
    p2 = dir_path / "execution_time.png"
    fig2.savefig(p2, dpi=300)
    print(f"Execution times plot saved to '{p2}'")
    plt.close(fig2)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        log_file_path = sys.argv[1]
    else:
        print("ERROR: Provide syslog file path")
        sys.exit(1)

    print(f"Processing: {log_file_path}")

    file_stem = Path(log_file_path).stem
    dir_path = Path(f'plot_{file_stem}')
    dir_path.mkdir(parents=True, exist_ok=True)

    parse_log_and_plot(log_file_path, dir_path)