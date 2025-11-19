import pandas as pd
import matplotlib.pyplot as plt
import os

ALL_STATS_CSV = "all_images_stats.csv"
SELECTED_CSV  = "selected_images_splits.csv"

# Directory for saving all plots
PLOT_DIR = "plots/distribution_bins"
os.makedirs(PLOT_DIR, exist_ok=True)


def savefig(name):
    """Helper to save figures to the output directory."""
    out_path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def plot_bin_counts(df_all, df_sel):
    counts_all = df_all["ratio_bin"].value_counts().sort_index()
    counts_sel = df_sel["ratio_bin"].value_counts().sort_index()

    print("Pre-subsampling counts:")
    print(counts_all)
    print("\nPost-subsampling counts:")
    print(counts_sel)

    bins = counts_all.index.tolist()
    x = range(len(bins))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], counts_all[bins], width=width, label="Before subsampling")
    plt.bar([i + width / 2 for i in x], counts_sel[bins], width=width, label="After subsampling")

    plt.xticks(list(x), bins)
    plt.ylabel("Number of images")
    plt.xlabel("Building-ratio bin")
    plt.title("Bin counts before and after subsampling")
    plt.legend()
    plt.tight_layout()

    savefig("bin_counts_before_after")
    plt.show()


def plot_histograms(df_all, df_sel):
    # Overall histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df_all["building_ratio"], bins=50, alpha=0.5, label="Before subsampling")
    plt.hist(df_sel["building_ratio"], bins=50, alpha=0.5, label="After subsampling")
    plt.xlabel("Building ratio")
    plt.ylabel("Number of images")
    plt.title("Building-ratio distribution (overall)")
    plt.legend()
    plt.tight_layout()

    savefig("hist_building_ratio_overall")
    plt.show()

    # Per-bin histogram
    for bname in sorted(df_all["ratio_bin"].unique()):
        plt.figure(figsize=(8, 5))
        r_all = df_all[df_all["ratio_bin"] == bname]["building_ratio"]
        r_sel = df_sel[df_sel["ratio_bin"] == bname]["building_ratio"]

        plt.hist(r_all, bins=30, alpha=0.5, label="Before subsampling")
        plt.hist(r_sel, bins=30, alpha=0.5, label="After subsampling")
        plt.xlabel("Building ratio")
        plt.ylabel("Number of images")
        plt.title(f"Building-ratio distribution in bin '{bname}'")
        plt.legend()
        plt.tight_layout()

        savefig(f"hist_ratio_bin_{bname}")
        plt.show()


def suggest_bin_edges(df_all, n_bins=4):
    ratios = df_all["building_ratio"]
    quantiles = [i / n_bins for i in range(n_bins + 1)]
    q_values = ratios.quantile(quantiles)

    print("\nSuggested bin edges based on quantiles:")
    for q, v in zip(quantiles, q_values):
        print(f"  q={q:.2f} -> {v:.6f}")

    # Save quantiles to a file for documentation
    out_path = os.path.join(PLOT_DIR, "suggested_bin_edges.txt")
    with open(out_path, "w") as f:
        f.write("Suggested bin edges (quantiles):\n")
        for q, v in zip(quantiles, q_values):
            f.write(f"q={q:.2f} -> {v:.6f}\n")

    print(f"Saved suggested bin edges to: {out_path}")

    return q_values.tolist()


def main():
    df_all = pd.read_csv(ALL_STATS_CSV)   # pre-subsampling
    df_sel = pd.read_csv(SELECTED_CSV)    # post-subsampling

    plot_bin_counts(df_all, df_sel)
    plot_histograms(df_all, df_sel)
    suggest_bin_edges(df_all, n_bins=4)


if __name__ == "__main__":
    main()
