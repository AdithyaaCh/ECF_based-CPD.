import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results", "known_k")
OUTDIR = os.path.join(ROOT, "comparative_plots")
os.makedirs(OUTDIR, exist_ok=True)


def _heat(ax, fig, data, row_vals, col_vals, cmap, vmin, vmax, title, cbar_label,
          xlabel, ylabel, fmt="{:.0f}"):
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels([f"{v:g}" for v in col_vals], fontsize=7)
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels([f"{v:g}" for v in row_vals], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    for i in range(len(row_vals)):
        for j in range(len(col_vals)):
            val = data[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="lightgrey"))
                continue
            color = "white" if val > (vmax * 0.6) else "black"
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=6, color=color)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(cbar_label, fontsize=9)


def _three_panel(ecf_df, midast_df, row_col, col_col, row_label, col_label, title, out_name):
    row_vals = sorted(ecf_df[row_col].unique())
    col_vals = sorted(ecf_df[col_col].unique())

    ecf_pivot = ecf_df.groupby([row_col, col_col])["MAE"].mean().unstack(col_col).reindex(index=row_vals, columns=col_vals)
    midast_mae = midast_df.groupby([row_col, col_col])["MAE"].mean().unstack(col_col).reindex(index=row_vals, columns=col_vals)
    midast_recall = (midast_df.assign(detected=midast_df["MAE"].notna())
                     .groupby([row_col, col_col])["detected"].mean().mul(100)
                     .unstack(col_col).reindex(index=row_vals, columns=col_vals))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    _heat(axes[0], fig, ecf_pivot.values, row_vals, col_vals, "RdYlGn_r", 0, 250,
          "ECF (always answers)", "MAE", col_label, row_label)
    _heat(axes[1], fig, midast_mae.values, row_vals, col_vals, "RdYlGn_r", 0, 250,
          "MIDAST[KS] (MAE | detected)", "MAE | detected", col_label, row_label)
    _heat(axes[2], fig, midast_recall.values, row_vals, col_vals, "RdYlGn", 0, 100,
          "MIDAST[KS] recall (%)", "Recall %", col_label, row_label)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTDIR, out_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_separate_dirs(ecf_dir, midast_dir, row_col, col_col, row_label, col_label, title, out_name):
    dfE = pd.read_csv(os.path.join(ecf_dir, "raw.csv"))
    dfM = pd.read_csv(os.path.join(midast_dir, "raw.csv"))
    _three_panel(dfE, dfM, row_col, col_col, row_label, col_label, title, out_name)


def plot_combined_method_col(csv_path, row_col, col_col, row_label, col_label, title, out_name):
    df = pd.read_csv(csv_path)
    ecf = df[df.Method == "ECF[argmax]"]
    midast = df[df.Method == "MIDAST[KS]"]
    _three_panel(ecf, midast, row_col, col_col, row_label, col_label, title, out_name)


def plot_timing(ecf_dir, midast_dir, title, out_name):
    dfE = pd.read_csv(os.path.join(ecf_dir, "raw.csv"))
    dfM = pd.read_csv(os.path.join(midast_dir, "raw.csv"))

    ecf_ms, ecf_std = dfE["time_s"].mean() * 1000, dfE["time_s"].std() * 1000
    mid_ms, mid_std = dfM["time_s"].mean() * 1000, dfM["time_s"].std() * 1000

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bars = ax.bar(["ECF", "MIDAST[KS]"], [ecf_ms, mid_ms], yerr=[ecf_std, mid_std],
                  color=["#2a9d8f", "#e76f51"], width=0.5, capsize=8, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("Mean runtime per series (ms, log scale)", fontsize=10)
    ax.set_title(title, fontsize=10)
    for b, v in zip(bars, [ecf_ms, mid_ms]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} ms", ha="center", va="bottom", fontsize=10)
    speedup = mid_ms / ecf_ms
    ax.text(0.5, 0.02, f"ECF is {speedup:.1f}x faster" if speedup >= 1 else f"MIDAST is {1/speedup:.1f}x faster",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTDIR, out_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    for dim in (2, 10):
        base = os.path.join(RESULTS, f"subgaussian_d{dim}")
        plot_separate_dirs(
            os.path.join(base, "ecf"), os.path.join(base, "midast"),
            row_col="alpha2", col_col="rho2",
            row_label=r"$\alpha_2$ (post-change tail index)",
            col_label=r"$\rho_2$ (post-change correlation)",
            title=f"Sub-Gaussian d={dim} | ECF vs MIDAST",
            out_name=f"subgaussian_d{dim}.png")
        plot_timing(os.path.join(base, "ecf"), os.path.join(base, "midast"),
                    title=f"Runtime -- Sub-Gaussian d={dim}", out_name=f"timing_subgaussian_d{dim}.png")

    for dim in (2, 10):
        base = os.path.join(RESULTS, f"lomax_d{dim}")
        plot_separate_dirs(
            os.path.join(base, "ecf"), os.path.join(base, "midast"),
            row_col="alpha1", col_col="alpha2",
            row_label=r"$\alpha_1$ (pre-change tail index)",
            col_label=r"$\alpha_2$ (post-change tail index)",
            title=f"Lomax (Pareto Type II) d={dim} | ECF vs MIDAST",
            out_name=f"lomax_d{dim}.png")
        plot_timing(os.path.join(base, "ecf"), os.path.join(base, "midast"),
                    title=f"Runtime -- Lomax d={dim}", out_name=f"timing_lomax_d{dim}.png")

    base_d2 = os.path.join(RESULTS, "studentt_d2")
    plot_separate_dirs(
        os.path.join(base_d2, "ecf"), os.path.join(base_d2, "midast"),
        row_col="nu2", col_col="rho2",
        row_label=r"$\nu_2$ (post-change degrees of freedom)",
        col_label=r"$\rho_2$ (post-change correlation)",
        title="Student-t d=2 | ECF vs MIDAST",
        out_name="studentt_d2.png")
    plot_timing(os.path.join(base_d2, "ecf"), os.path.join(base_d2, "midast"),
                title="Runtime -- Student-t d=2", out_name="timing_studentt_d2.png")

    d10_csv = os.path.join(RESULTS, "studentt_d10", "raw.csv")
    if os.path.exists(d10_csv):
        plot_combined_method_col(
            d10_csv, row_col="nu2", col_col="rho2",
            row_label=r"$\nu_2$ (post-change degrees of freedom)",
            col_label=r"$\rho_2$ (post-change correlation)",
            title="Student-t d=10 | ECF vs MIDAST",
            out_name="studentt_d10.png")
    else:
        print(f"[skip] {d10_csv} not found -- run run_studentt.py --dim 10 first")


if __name__ == "__main__":
    main()
