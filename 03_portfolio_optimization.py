"""
03_portfolio_optimization.py — Intra-Cluster Offshore Wind Portfolio Optimization
===================================================================================
For each regional cluster of prime offshore wind sites, finds subclusters with
distinct (ideally anti-correlated) monthly WPD profiles, then runs Markowitz
mean-variance optimisation within the cluster.

Usage:  python 03_portfolio_optimization.py

Inputs:
    data/suitability_results_2025.parquet   ← golden-zone cells
    data/era5_month_01.parquet … _12.parquet ← monthly ERA5 physics

Outputs:
    data/portfolio_results.parquet
    outputs/plots/<region>_subclusters.png
    outputs/plots/<region>_corr_matrix.png
    outputs/plots/<region>_seasonal.png
    outputs/plots/<region>_frontier.png
    outputs/plots/cluster_summary.png
"""

import sys
import traceback
import warnings
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ───────────────────────────────────────────────────────────────────
SUIT_PQ   = Path("data/suitability_results_2025.parquet")
OUT_DIR   = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PQ = Path("data/portfolio_results.parquet")

MONTHS = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MIN_REGION_CELLS = 30  # need enough cells to subcluster

# Region definitions: name → (lat_min, lat_max, lon_min, lon_max)
REGION_DEFS = {
    "Alaska":         (54.0,  73.0, -160.0, -140.0),
    "Pacific":        (32.0,  49.0, -130.0, -117.0),
    "Gulf of Mexico": (24.0,  31.0,  -98.0,  -82.0),
    "Southeast":      (25.0,  38.0,  -82.0,  -74.0),
    "Mid-Atlantic":   (38.0,  43.0,  -76.0,  -64.0),
    "New England":    (43.0,  46.0,  -72.0,  -64.0),
}

REGION_COLORS_MAP = {
    "Alaska":         "#1f77b4",
    "Pacific":        "#2ca02c",
    "Gulf of Mexico": "#d62728",
    "Southeast":      "#ff7f0e",
    "Mid-Atlantic":   "#9467bd",
    "New England":    "#8c564b",
}

SUBCLUSTER_CMAP = plt.cm.Set2

# Number of subclusters to find within each region
N_SUBCLUSTERS = 3


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Assign golden-zone cells to regions
# ─────────────────────────────────────────────────────────────────────────────

def assign_regions(suit_pq: Path) -> pd.DataFrame:
    """Load golden-zone cells and assign each to a coastal region."""
    print("[1] Assigning golden-zone cells to regions ...")
    df = pd.read_parquet(suit_pq)

    def _region(lat, lon):
        for name, (la_min, la_max, lo_min, lo_max) in REGION_DEFS.items():
            if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
                return name
        return None

    df["region"] = df.apply(lambda r: _region(r["lat"], r["lon"]), axis=1)
    df = df.dropna(subset=["region"])

    for name, grp in sorted(df.groupby("region"), key=lambda x: -len(x[1])):
        print(f"    {name:20s}: {len(grp):5d} cells")

    # Drop regions with too few cells
    counts = df["region"].value_counts()
    keep = counts[counts >= MIN_REGION_CELLS].index.tolist()
    dropped = [r for r in counts.index if r not in keep]
    if dropped:
        print(f"    Dropping sparse regions: {dropped}")
    df = df[df["region"].isin(keep)].reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Extract per-cell monthly WPD for each region
# ─────────────────────────────────────────────────────────────────────────────

def extract_cell_monthly_wpd(golden_df: pd.DataFrame) -> dict:
    """
    For each region, extract the 12-month WPD time series for every
    individual golden-zone cell.

    Returns:
        dict[region_name] → DataFrame with columns:
            lat, lon, month_1, month_2, ..., month_12
    """
    print("\n[2] Extracting per-cell monthly WPD ...")

    golden_coords = golden_df[["lat", "lon", "region"]].copy()
    golden_coords["lat_snap"] = (golden_coords["lat"] / 0.25).round() * 0.25
    golden_coords["lon_snap"] = (golden_coords["lon"] / 0.25).round() * 0.25
    tmp_golden = Path("/tmp/_golden_coords.parquet")
    golden_coords.to_parquet(tmp_golden, index=False)

    con = duckdb.connect()

    # Pull all months in one pass per month, then pivot
    all_records = []
    for month in MONTHS:
        month_pq = Path(f"data/era5_month_{month:02d}.parquet")
        if not month_pq.exists():
            print(f"    Month {month:2d}: MISSING")
            continue

        sql = f"""
        SELECT
            g.lat, g.lon, g.region,
            e.wpd_mean AS wpd
        FROM read_parquet('{tmp_golden}') g
        INNER JOIN read_parquet('{month_pq}') e
            ON ROUND(e.lat / 0.25) * 0.25 = g.lat_snap
           AND ROUND(e.lon / 0.25) * 0.25 = g.lon_snap
        """
        result = con.execute(sql).df()
        result["month"] = month
        all_records.append(result)

    con.close()
    tmp_golden.unlink(missing_ok=True)

    all_data = pd.concat(all_records, ignore_index=True)

    # Pivot: each row = one cell (lat, lon), columns = month_1...month_12
    region_data = {}
    for region in golden_df["region"].unique():
        rdf = all_data[all_data["region"] == region].copy()
        pivot = rdf.pivot_table(
            index=["lat", "lon"], columns="month", values="wpd",
        )
        pivot.columns = [f"month_{m}" for m in pivot.columns]
        pivot = pivot.dropna().reset_index()

        region_data[region] = pivot
        print(f"    {region:20s}: {len(pivot)} cells × 12 months")

    return region_data


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Correlation-based subclustering within each region
# ─────────────────────────────────────────────────────────────────────────────

def find_subclusters(region_data: dict, n_clusters: int = N_SUBCLUSTERS) -> dict:
    """
    Within each region, cluster cells by the similarity of their monthly WPD
    profiles using hierarchical clustering on correlation distance.

    Returns:
        dict[region] → DataFrame with added 'subcluster' column
    """
    print(f"\n[3] Finding {n_clusters} subclusters per region ...")

    results = {}
    for region, df in region_data.items():
        month_cols = [c for c in df.columns if c.startswith("month_")]
        X = df[month_cols].values  # (n_cells, 12)

        if len(df) < n_clusters:
            print(f"    {region}: too few cells ({len(df)}), skipping")
            continue

        # Correlation matrix between cells
        corr = np.corrcoef(X)
        # Clip to avoid numerical issues
        corr = np.clip(corr, -1, 1)

        # Convert correlation to distance: d = 1 - corr
        dist = 1.0 - corr
        np.fill_diagonal(dist, 0)
        # Ensure symmetry
        dist = (dist + dist.T) / 2
        dist = np.clip(dist, 0, None)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")

        # Cut into n_clusters
        actual_n = min(n_clusters, len(df))
        labels = fcluster(Z, t=actual_n, criterion="maxclust")

        df = df.copy()
        df["subcluster"] = labels - 1  # 0-indexed

        # Print subcluster stats
        for sc in sorted(df["subcluster"].unique()):
            sub_df = df[df["subcluster"] == sc]
            mean_wpd = sub_df[month_cols].mean(axis=1).mean()
            lat_range = f"{sub_df['lat'].min():.1f}-{sub_df['lat'].max():.1f}"
            lon_range = f"{sub_df['lon'].min():.1f}-{sub_df['lon'].max():.1f}"
            print(f"    {region} SC-{sc}: {len(sub_df):3d} cells, "
                  f"lat {lat_range}, lon {lon_range}, "
                  f"mean WPD = {mean_wpd:.0f} W/m²")

        results[region] = df

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Subcluster correlation & portfolio analysis per region
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(w, mu, cov):
    """(mean, std) for given weight vector."""
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def min_variance_weights(mu, cov):
    """Global minimum-variance portfolio (long only, weights sum to 1)."""
    n = len(mu)
    result = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result.x


def analyse_cluster(region: str, df: pd.DataFrame) -> dict:
    """
    For one region: compute subcluster monthly profiles, correlation,
    and optimal portfolio.

    Returns dict with all analysis results.
    """
    month_cols = [c for c in df.columns if c.startswith("month_")]
    subclusters = sorted(df["subcluster"].unique())
    n_sc = len(subclusters)

    # Compute subcluster mean monthly WPD profiles
    profiles = np.zeros((n_sc, 12))
    for i, sc in enumerate(subclusters):
        sub_df = df[df["subcluster"] == sc]
        profiles[i] = sub_df[month_cols].mean().values

    # Subcluster correlation & covariance
    corr = np.corrcoef(profiles)
    mu = profiles.mean(axis=1)     # mean WPD per subcluster
    cov = np.cov(profiles)         # covariance across months

    # Min-variance portfolio
    w_opt = min_variance_weights(mu, cov)
    port_ret, port_std = portfolio_stats(w_opt, mu, cov)

    # Compare to best single subcluster
    best_sc = np.argmax(mu)
    best_std = float(np.sqrt(cov[best_sc, best_sc]))
    var_reduction = (1.0 - port_std / best_std) * 100 if best_std > 0 else 0

    # Also compare to equal-weight portfolio
    w_equal = np.ones(n_sc) / n_sc
    eq_ret, eq_std = portfolio_stats(w_equal, mu, cov)

    print(f"\n    === {region} Portfolio Analysis ===")
    print(f"    Subcluster correlations:")
    for i in range(n_sc):
        for j in range(i + 1, n_sc):
            print(f"      SC-{i} ↔ SC-{j}: ρ = {corr[i, j]:.3f}")

    print(f"    Min-variance allocation: "
          + ", ".join(f"SC-{sc}={w_opt[i]*100:.1f}%" for i, sc in enumerate(subclusters)))
    print(f"    Portfolio: mean={port_ret:.0f} W/m², std={port_std:.0f} W/m²")
    print(f"    Best single subcluster (SC-{best_sc}): "
          f"mean={mu[best_sc]:.0f}, std={best_std:.0f}")
    print(f"    Equal-weight: mean={eq_ret:.0f}, std={eq_std:.0f}")
    print(f"    Variance reduction vs best single: {var_reduction:.1f}%")

    return {
        "region": region,
        "profiles": profiles,
        "corr": corr,
        "mu": mu,
        "cov": cov,
        "w_opt": w_opt,
        "port_ret": port_ret,
        "port_std": port_std,
        "best_single_std": best_std,
        "best_single_mean": mu[best_sc],
        "var_reduction": var_reduction,
        "subclusters": subclusters,
        "eq_ret": eq_ret,
        "eq_std": eq_std,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Efficient frontier within a cluster
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficient_frontier(mu, cov, n_points=80):
    n = len(mu)
    target_returns = np.linspace(np.min(mu), np.max(mu), n_points)
    frontier_rets, frontier_stds = [], []

    for tr in target_returns:
        result = minimize(
            lambda w: w @ cov @ w,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "ineq", "fun": lambda w, t=tr: w @ mu - t},
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        r, s = portfolio_stats(result.x, mu, cov)
        frontier_rets.append(r)
        frontier_stds.append(s)

    return np.array(frontier_rets), np.array(frontier_stds)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(region: str) -> str:
    return region.lower().replace(" ", "_").replace("/", "_")


def plot_subcluster_map(region: str, df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter plot of cells coloured by subcluster."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    for sc in sorted(df["subcluster"].unique()):
        sub = df[df["subcluster"] == sc]
        color = SUBCLUSTER_CMAP(sc / max(df["subcluster"].max(), 1))
        ax.scatter(sub["lon"], sub["lat"], s=30, alpha=0.8,
                   label=f"Subcluster {sc} ({len(sub)} cells)",
                   color=color, edgecolors="black", linewidths=0.3)

    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)", fontsize=11)
    ax.set_title(f"{region} — Subclusters by Monthly WPD Profile Similarity",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{_safe_name(region)}_subclusters.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def plot_seasonal_profiles(region: str, analysis: dict, out_dir: Path) -> None:
    """Line chart of monthly WPD per subcluster."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    profiles = analysis["profiles"]
    for i, sc in enumerate(analysis["subclusters"]):
        color = SUBCLUSTER_CMAP(sc / max(len(analysis["subclusters"]) - 1, 1))
        ax.plot(MONTHS, profiles[i], marker="o", markersize=5,
                linewidth=2.2, label=f"SC-{sc} (mean {analysis['mu'][i]:.0f} W/m²)",
                color=color)

    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Mean WPD (W/m²)", fontsize=11)
    ax.set_title(f"{region} — Monthly WPD by Subcluster",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{_safe_name(region)}_seasonal.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def plot_correlation_matrix(region: str, analysis: dict, out_dir: Path) -> None:
    """Heatmap of subcluster correlation."""
    corr = analysis["corr"]
    n = corr.shape[0]
    labels = [f"SC-{sc}" for sc in analysis["subclusters"]]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson ρ")
    ax.set_title(f"{region} — Subcluster WPD Correlation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / f"{_safe_name(region)}_corr_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def plot_frontier(region: str, analysis: dict, out_dir: Path) -> None:
    """Efficient frontier with subclusters and optimal portfolio."""
    mu, cov = analysis["mu"], analysis["cov"]
    w_opt = analysis["w_opt"]

    frontier_rets, frontier_stds = compute_efficient_frontier(mu, cov)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    # Frontier
    ax.plot(frontier_stds, frontier_rets, linewidth=2.5, color="#1f77b4",
            label="Efficient Frontier", zorder=3)

    # Individual subclusters
    for i, sc in enumerate(analysis["subclusters"]):
        std_i = float(np.sqrt(cov[i, i]))
        color = SUBCLUSTER_CMAP(sc / max(len(analysis["subclusters"]) - 1, 1))
        ax.scatter(std_i, mu[i], s=120, color=color, edgecolors="black",
                   linewidths=1.2, zorder=5, label=f"SC-{sc}")

    # Optimal portfolio
    ax.scatter(analysis["port_std"], analysis["port_ret"],
               s=200, marker="*", color="gold", edgecolors="black",
               linewidths=1.5, zorder=6,
               label=f"Min-Var Portfolio (σ↓{analysis['var_reduction']:.0f}%)")

    ax.set_xlabel("Risk (Std Dev of WPD, W/m²)", fontsize=11)
    ax.set_ylabel("Return (Mean WPD, W/m²)", fontsize=11)
    ax.set_title(f"{region} — Efficient Frontier (Intra-Cluster)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{_safe_name(region)}_frontier.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def plot_summary(all_analyses: list, out_dir: Path) -> None:
    """Summary bar chart: variance reduction per region from intra-cluster diversification."""
    regions = [a["region"] for a in all_analyses]
    var_reductions = [a["var_reduction"] for a in all_analyses]
    colors = [REGION_COLORS_MAP.get(r, "#666") for r in regions]

    # Find minimum subcluster correlation per region
    min_corrs = []
    for a in all_analyses:
        corr = a["corr"]
        n = corr.shape[0]
        off_diag = [corr[i, j] for i in range(n) for j in range(i+1, n)]
        min_corrs.append(min(off_diag) if off_diag else 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Variance reduction
    bars = ax1.bar(regions, var_reductions, color=colors,
                   edgecolor="white", linewidth=1.2)
    for bar, vr in zip(bars, var_reductions):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{vr:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax1.set_ylabel("Variance Reduction vs Best Single Subcluster (%)", fontsize=10)
    ax1.set_title("Diversification Benefit Per Region", fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30, labelsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Minimum subcluster correlation
    bars2 = ax2.bar(regions, min_corrs, color=colors,
                    edgecolor="white", linewidth=1.2)
    for bar, mc in zip(bars2, min_corrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{mc:.2f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_ylabel("Minimum Subcluster Correlation (ρ)", fontsize=10)
    ax2.set_title("Lowest Within-Cluster Correlation", fontsize=13, fontweight="bold")
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="ρ = 0")
    ax2.tick_params(axis="x", rotation=30, labelsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "cluster_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n    Saved summary → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_analyses: list, clustered_data: dict,
                 result_pq: Path) -> None:
    """Save portfolio results to parquet."""
    print(f"\n[7] Saving results → {result_pq}")

    rows = []
    for a in all_analyses:
        for i, sc in enumerate(a["subclusters"]):
            rows.append({
                "region": a["region"],
                "subcluster": sc,
                "mean_wpd": a["mu"][i],
                "std_wpd": float(np.sqrt(a["cov"][i, i])),
                "optimal_weight_pct": a["w_opt"][i] * 100,
                "portfolio_mean_wpd": a["port_ret"],
                "portfolio_std_wpd": a["port_std"],
                "var_reduction_pct": a["var_reduction"],
            })

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(result_pq, index=False)
    print(result_df.round(1).to_string(index=False))

    # Also save the full cell-level data with subcluster assignments
    all_cells = []
    for region, df in clustered_data.items():
        all_cells.append(df)
    if all_cells:
        cell_df = pd.concat(all_cells, ignore_index=True)
        cell_path = Path("data/portfolio_cell_assignments.parquet")
        cell_df.to_parquet(cell_path, index=False)
        print(f"    Cell assignments → {cell_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        # 1. Region assignment
        golden = assign_regions(SUIT_PQ)

        # 2. Per-cell monthly WPD
        region_data = extract_cell_monthly_wpd(golden)

        # 3. Subclustering
        clustered = find_subclusters(region_data)

        # 4. Analysis per region
        print("\n[4] Portfolio analysis per region ...")
        all_analyses = []
        for region, df in clustered.items():
            analysis = analyse_cluster(region, df)
            all_analyses.append(analysis)

        # 5-6. Plots
        print("\n[5] Generating plots ...")
        for region, df in clustered.items():
            analysis = [a for a in all_analyses if a["region"] == region][0]
            plot_subcluster_map(region, df, OUT_DIR)
            plot_seasonal_profiles(region, analysis, OUT_DIR)
            plot_correlation_matrix(region, analysis, OUT_DIR)
            plot_frontier(region, analysis, OUT_DIR)

        plot_summary(all_analyses, OUT_DIR)

        # 7. Save
        save_results(all_analyses, clustered, RESULT_PQ)

        print(f"\n[Done] All outputs written.")
        print(f"  Plots   : {OUT_DIR}/")
        print(f"  Results : {RESULT_PQ}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
