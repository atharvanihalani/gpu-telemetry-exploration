"""Small-multiples gallery: does temperature follow power in every recording?

For each condition, plot node-mean power (gray) and temp (blue) over the FULL
run (including startup, where the power step makes the thermal chase visible),
with real units on twin y-axes: power (W) on the left, temp (°C) on the right.
Annotate with the plain Pearson correlation of the binned series.

Session 12 finding: r = 0.73-0.98 across all 20 conditions; low-r panels are
exactly the ones with negligible power swing (nothing to follow).

Usage:  python analysis/temp_follows_power.py
Output: plots/12_temp_follows_power.png
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
NPTS = 80  # time bins per panel

CONDITIONS = [
    ("t1", "T1 · DDP 3.37B"), ("t2", "T2 · DDP 136M"), ("t3", "T3 · grad accum"),
    ("t4", "T4 · pipeline"), ("t5", "T5 · grad ckpt"), ("t6", "T6 · FSDP"),
    ("i2", "I2 · autoregressive"), ("i3", "I3 · vLLM TP"), ("i4", "I4 · spec decode"),
    ("e2", "E2 · cover traffic"), ("e3", "E3 · intermittent"), ("e4", "E4 · PCIe allreduce"),
    ("e5", "E5 · smoothed"), ("b1", "B1 · idle baseline"),
    ("t10_node0", "T10 · 16-GPU DDP"), ("t11_node0", "T11 · TP+DP"),
    ("t12_node0", "T12 · MoE EP+DP"), ("t13_node0", "T13 · TP+PP"),
    ("t14_node0", "T14 · TP+EP+DP"), ("t15_node0", "T15 · FSDP-16"),
]


def load_binned(key):
    """Node-mean power/temp for the full run, averaged into NPTS time bins."""
    df = pd.read_csv(os.path.join(ROOT, "data", f"{key}_telemetry.csv"))
    ts = df.groupby("timestamp")[["power_w", "temp_c"]].mean().reset_index().sort_values("timestamp")
    bins = np.linspace(ts.timestamp.min(), ts.timestamp.max(), NPTS + 1)
    ts["bin"] = pd.cut(ts.timestamp, bins, labels=False, include_lowest=True)
    b = ts.groupby("bin")[["power_w", "temp_c"]].mean()
    dur = ts.timestamp.max() - ts.timestamp.min()
    return b.power_w.values, b.temp_c.values, dur


def main():
    fig, axes = plt.subplots(5, 4, figsize=(18, 14))
    fig.suptitle("Temperature follows power — every recorded condition\n"
                 "gray = power (W, left axis, fixed 0-700W), blue = temp (°C, right axis, fixed 25-70°C)",
                 fontsize=14, y=0.995)

    for ax, (key, label) in zip(axes.flat, CONDITIONS):
        P, T, dur = load_binned(key)
        r = np.corrcoef(P, T)[0, 1]
        x = np.linspace(0, dur / 60, len(P))
        axt = ax.twinx()
        lp, = ax.plot(x, P, color="#898781", lw=1.4, label="power (W)")
        lt, = axt.plot(x, T, color="#2a78d6", lw=1.4, label="temp (°C)")
        ax.set_ylim(0, 700)  # common scale: H100 TDP
        axt.set_ylim(25, 70)  # common temp scale across panels
        ax.set_title(f"{label}   r={r:+.2f}", fontsize=10)
        ax.text(0.98, 0.04, f"ΔP={P.max()-P.min():.0f}W",
                transform=ax.transAxes, ha="right", fontsize=8, color="#888")
        ax.tick_params(labelsize=8, colors="#898781", labelcolor="#66645f")
        axt.tick_params(labelsize=8, colors="#2a78d6", labelcolor="#2a78d6")
        ax.tick_params(axis="x", colors="black", labelcolor="black")
        ax.set_xlabel("min", fontsize=8, labelpad=1)
        if ax is axes.flat[0]:
            ax.legend(handles=[lp, lt], fontsize=8, loc="lower right")

    fig.tight_layout()
    out = os.path.join(ROOT, "plots", "12_temp_follows_power.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    rs = []
    for key, label in CONDITIONS:
        P, T, _ = load_binned(key)
        rs.append((label, np.corrcoef(P, T)[0, 1], P.max() - P.min()))
    rs.sort(key=lambda x: x[1])
    print("\ncorrelation summary (sorted):")
    for label, r, dp in rs:
        print(f"  {label:24s} r={r:+.2f}  (power swing {dp:.0f}W)")
    print(f"\nmedian r = {np.median([r for _, r, _ in rs]):.2f}")


if __name__ == "__main__":
    main()
