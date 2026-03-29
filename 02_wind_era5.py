"""
02_wind_era5.py — ERA5 Wind Physics, Suitability Model & Cluster Analysis
==========================================================================

Usage:
  python 02_wind_era5.py month <N>       Process month N (1-12) → monthly parquet
  python 02_wind_era5.py aggregate       Combine monthly parquets → suitability

Source  : data/Wind/oper*.nc, data/Wind/wave*.nc  (ERA5 NetCDF4)

File → month mapping:
  oper2/wave2 → Jan-Mar   (u100/v100 in oper file)
  oper3/wave3 → Apr-May   (u100/v100 in oper file)
  oper4/wave4 → Jun-Jul   (u100/v100 in oper file)
  oper1/wave1 → Nov-Dec   (u100/v100 from missing_wind_vectors.nc)
  oper5/wave5 → Aug-Oct   (u100/v100 from missing_wind_vectors.nc)

Outputs (month mode):
  data/era5_month_01.parquet … data/era5_month_12.parquet

Outputs (aggregate mode):
  data/era5_annual_means_2025.parquet   — gridded annual-mean physics fields
  data/suitability_results_2025.parquet — spatial join with bathymetry (Golden Zones)
  data/prime_wind_sites.parquet         — top 10% suitability + DBSCAN clusters
  outputs/plots/wpd_heatmap.png
  outputs/plots/golden_zones.png
"""

import sys
import traceback
import warnings
import gc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import xarray as xr
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

try:
    import cmocean
    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False

# -- Config --------------------------------------------------------------------
WIND_DIR   = Path("data/Wind")
BATHY_PQ   = Path("data/bathymetry_processed.parquet")
ERA5_PQ    = Path("data/era5_annual_means_2025.parquet")
RESULT_PQ  = Path("data/suitability_results_2025.parquet")
PRIME_PQ   = Path("data/prime_wind_sites.parquet")
OUT_DIR    = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2025

# Map file index → months contained
FILE_MONTHS: dict[int, list[int]] = {
    1: [11, 12],        # Nov-Dec
    2: [1, 2, 3],       # Jan-Mar
    3: [4, 5],          # Apr-May  (note: ends May 30)
    4: [6, 7],          # Jun-Jul
    5: [8, 9, 10],      # Aug-Oct
}

# File indices that have u100/v100 in the oper file itself
FILES_WITH_WIND = {2, 3, 4}

# Supplemental wind vectors for months missing from oper files (Aug-Dec)
WIND_SUPPLEMENT = WIND_DIR / "missing_wind_vectors.nc"

# Reverse lookup: month → file index
MONTH_TO_FILE: dict[int, int] = {}
for fidx, months in FILE_MONTHS.items():
    for m in months:
        MONTH_TO_FILE[m] = fidx

# Golden Zone thresholds (industry-standard for offshore wind)
WPD_MIN_W_M2 = 200.0     # IEC Class III minimum (~6 m/s mean)
DEPTH_MIN_M  = 10.0      # Monopile minimum practical depth
DEPTH_MAX_M  = 1000.0    # Include deep-water floating (semi-sub, TLP)
SWH_P90_MAX  = 5.0       # 90th-percentile operability limit (not extreme max)
SWH_MAX_HARD = 14.0      # Absolute structural survival limit

# Physical constants
R_DRY_AIR = 287.058  # J/(kg*K)

# Suitability weights
W_WPD   = 0.5
W_DEPTH = 0.3
W_WAVE  = 0.2

# DBSCAN parameters
DBSCAN_EPS     = 0.75
DBSCAN_MIN_PTS = 2


def _month_parquet_path(month: int) -> Path:
    return Path(f"data/era5_month_{month:02d}.parquet")


# ------------------------------------------------------------------------------
# SECTION 1 — Read NetCDF files via xarray
# ------------------------------------------------------------------------------

def _process_month_from_nc(
    month: int,
) -> tuple[xr.Dataset, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Read ERA5 NetCDF files for the given month, compute wind physics.

    Returns:
        (physics_ds, lsm_grid, lsm_lats, lsm_lons)
    """
    fidx = MONTH_TO_FILE[month]
    oper_path = WIND_DIR / f"oper{fidx}.nc"
    wave_path = WIND_DIR / f"wave{fidx}.nc"

    print(f"[1] Reading NetCDF for {YEAR}-{month:02d} ...")
    print(f"    Oper file: {oper_path}")
    print(f"    Wave file: {wave_path}")

    # -- Load oper file and filter to target month --
    oper_ds = xr.open_dataset(oper_path)
    oper_month = oper_ds.sel(valid_time=oper_ds.valid_time.dt.month == month)

    sfc_lats = oper_ds.latitude.values
    sfc_lons = oper_ds.longitude.values

    # -- Land-sea mask (time-invariant, take first timestep) --
    lsm_grid = None
    lsm_lats = None
    lsm_lons = None
    if "lsm" in oper_ds.data_vars:
        lsm_grid = oper_ds["lsm"].isel(valid_time=0).values
        lsm_lats = sfc_lats
        lsm_lons = sfc_lons
        print(f"    Land-sea mask: {lsm_grid.shape}")

    # -- Load wind vectors (from oper file or supplement) --
    if "u100" in oper_month.data_vars:
        u100 = oper_month["u100"].values  # (time, lat, lon)
        v100 = oper_month["v100"].values
    else:
        print(f"    u100/v100 not in oper file, reading from {WIND_SUPPLEMENT.name} ...")
        wind_ds_sup = xr.open_dataset(WIND_SUPPLEMENT)
        wind_month = wind_ds_sup.sel(valid_time=wind_ds_sup.valid_time.dt.month == month)
        u100 = wind_month["u100"].values
        v100 = wind_month["v100"].values
        wind_ds_sup.close()
        del wind_month

    # -- Compute wind physics --
    t2m  = oper_month["t2m"].values
    sp   = oper_month["sp"].values

    n_timesteps = u100.shape[0]
    print(f"    Timesteps for month {month}: {n_timesteps}")

    ws  = np.sqrt(u100 ** 2 + v100 ** 2)
    rho = sp / (R_DRY_AIR * t2m)
    wpd = 0.5 * rho * ws ** 3

    # Mean over time axis
    wpd_mean = np.nanmean(wpd, axis=0).astype(np.float32)
    ws_mean  = np.nanmean(ws,  axis=0).astype(np.float32)
    rho_mean = np.nanmean(rho, axis=0).astype(np.float32)

    wind_ds = xr.Dataset(
        {
            "wpd_mean": (["lat", "lon"], wpd_mean),
            "ws_mean":  (["lat", "lon"], ws_mean),
            "rho_mean": (["lat", "lon"], rho_mean),
        },
        coords={"lat": sfc_lats, "lon": sfc_lons},
    )

    del u100, v100, t2m, sp, ws, rho, wpd, wpd_mean, ws_mean, rho_mean
    oper_ds.close()
    del oper_month
    gc.collect()
    print("    Wind physics done.", flush=True)

    # -- Wave data --
    if wave_path.exists():
        wave_ds_raw = xr.open_dataset(wave_path)
        wave_month = wave_ds_raw.sel(
            valid_time=wave_ds_raw.valid_time.dt.month == month,
        )

        wave_lats = wave_ds_raw.latitude.values
        wave_lons = wave_ds_raw.longitude.values

        swh = wave_month["swh"].values  # (time, lat, lon)
        mwp = wave_month["mwp"].values if "mwp" in wave_month else None

        wave_ds = xr.Dataset(
            {
                "swh_p90":  (["lat", "lon"], np.nanquantile(swh, 0.90, axis=0).astype(np.float32)),
                "swh_max":  (["lat", "lon"], np.nanmax(swh, axis=0).astype(np.float32)),
                "mwp_mean": (["lat", "lon"],
                             np.nanmean(mwp, axis=0).astype(np.float32)
                             if mwp is not None
                             else np.full(swh.shape[1:], np.nan, dtype=np.float32)),
            },
            coords={"lat": wave_lats, "lon": wave_lons},
        )
        del swh, mwp
        wave_ds_raw.close()
        del wave_month
        gc.collect()

        # Interpolate wave 0.5° → 0.25° to match surface grid
        wave_ds = wave_ds.interp(lat=sfc_lats, lon=sfc_lons, method="linear")
        print("    Wave stats done.", flush=True)

        physics_ds = xr.merge([wind_ds, wave_ds])
        del wind_ds, wave_ds
    else:
        print("    WARNING: No wave file found — filling NaN.")
        physics_ds = wind_ds
        shape = (len(sfc_lats), len(sfc_lons))
        physics_ds["swh_p90"]  = xr.DataArray(np.full(shape, np.nan, dtype=np.float32), dims=["lat", "lon"])
        physics_ds["swh_max"]  = xr.DataArray(np.full(shape, np.nan, dtype=np.float32), dims=["lat", "lon"])
        physics_ds["mwp_mean"] = xr.DataArray(np.full(shape, np.nan, dtype=np.float32), dims=["lat", "lon"])

    gc.collect()
    return physics_ds, lsm_grid, lsm_lats, lsm_lons


# ------------------------------------------------------------------------------
# SECTION 2 — Distance to Shore from land-sea mask
# ------------------------------------------------------------------------------

def compute_distance_to_shore(
    lsm_grid: np.ndarray,
    lsm_lats: np.ndarray,
    lsm_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    """
    Compute approximate distance-to-shore (km) from land-sea mask.
    Interpolates to target grid if grids differ.
    """
    print("    Computing distance to shore from land-sea mask ...")
    land_binary = (lsm_grid > 0.5).astype(np.uint8)
    dist_cells = distance_transform_edt(1 - land_binary)

    lat_grid = np.abs(np.diff(lsm_lats).mean())
    km_per_cell = lat_grid * 111.0
    dist_km = (dist_cells * km_per_cell).astype(np.float32)

    # If grids match, return directly; otherwise interpolate
    if np.array_equal(lsm_lats, target_lats) and np.array_equal(lsm_lons, target_lons):
        print(f"    Distance range: {dist_km.min():.1f} - {dist_km.max():.1f} km")
        return dist_km

    da = xr.DataArray(
        dist_km,
        dims=["lat", "lon"],
        coords={"lat": lsm_lats, "lon": lsm_lons},
    )
    da_interp = da.interp(lat=target_lats, lon=target_lons, method="nearest")
    result = da_interp.values
    print(f"    Distance range: {np.nanmin(result):.1f} - {np.nanmax(result):.1f} km")
    return result


# ------------------------------------------------------------------------------
# SECTION 3 — "month" mode: read NetCDF → parquet
# ------------------------------------------------------------------------------

def run_single_month(month: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Processing month {month} of {YEAR}")
    print(f"{'='*60}\n")

    # 1. Read NetCDF
    physics_ds, lsm_grid, lsm_lats, lsm_lons = _process_month_from_nc(month)

    # 2. Distance to shore
    print(f"\n[2] Distance to shore ...")
    sfc_lats = physics_ds.lat.values
    sfc_lons = physics_ds.lon.values

    if lsm_grid is not None:
        dist_km = compute_distance_to_shore(
            lsm_grid, lsm_lats, lsm_lons, sfc_lats, sfc_lons,
        )
    else:
        print("    WARNING: No land-sea mask found — filling distance with NaN.")
        dist_km = np.full((len(sfc_lats), len(sfc_lons)), np.nan, dtype=np.float32)

    # 3. Apply land-sea mask and flatten to DataFrame
    print(f"\n[3] Exporting parquet ...")
    lat2d, lon2d = np.meshgrid(sfc_lats, sfc_lons, indexing="ij")

    # Build ocean mask from land-sea mask (lsm <= 0.5 = ocean)
    if lsm_grid is not None:
        ocean_mask = (lsm_grid <= 0.5).ravel()
    else:
        ocean_mask = np.ones(lat2d.size, dtype=bool)

    df = pd.DataFrame({
        "lat":              lat2d.ravel(),
        "lon":              lon2d.ravel(),
        "wpd_mean":         physics_ds["wpd_mean"].values.ravel(),
        "ws_mean":          physics_ds["ws_mean"].values.ravel(),
        "rho_mean":         physics_ds["rho_mean"].values.ravel(),
        "swh_p90":          physics_ds["swh_p90"].values.ravel(),
        "swh_max":          physics_ds["swh_max"].values.ravel(),
        "mwp_mean":         physics_ds["mwp_mean"].values.ravel(),
        "dist_to_shore_km": dist_km.ravel(),
    })

    n_before = len(df)
    df = df[ocean_mask].dropna(subset=["wpd_mean"])
    print(f"    Filtered {n_before - len(df):,} land cells (lsm > 0.5)")

    df["lat"] = df["lat"].round(4)
    df["lon"] = df["lon"].round(4)

    del physics_ds, lsm_grid, dist_km
    gc.collect()

    out_path = _month_parquet_path(month)
    df.to_parquet(out_path, index=False)
    print(f"\n    Saved -> {out_path}")
    print(f"    Rows: {len(df):,}  |  File: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"    WPD range: {df['wpd_mean'].min():.1f} - {df['wpd_mean'].max():.1f} W/m2")
    print(f"\n[Done] Month {month} complete.")


# ------------------------------------------------------------------------------
# SECTION 4 — "aggregate" mode: combine monthly parquets → suitability
# ------------------------------------------------------------------------------

def run_aggregate() -> None:
    print(f"\n{'='*60}")
    print(f"  Aggregating monthly parquets → Suitability Analysis")
    print(f"{'='*60}\n")

    available = []
    for m in range(1, 13):
        p = _month_parquet_path(m)
        if p.exists():
            available.append((m, p))

    if not available:
        print("ERROR: No monthly parquets found! Run 'month <N>' first.")
        sys.exit(1)

    months_found = [m for m, _ in available]
    print(f"[1] Found {len(available)}/12 monthly parquets: {months_found}")
    if len(available) < 12:
        missing = [m for m in range(1, 13) if m not in months_found]
        print(f"    Missing months: {missing}")
        print(f"    Proceeding with available data ...\n")

    print("[2] Computing annual means ...")
    frames = []
    for m, p in available:
        df = pd.read_parquet(p)
        df["_month"] = m
        frames.append(df)
        print(f"    Month {m:2d}: {len(df):,} rows, "
              f"WPD {df['wpd_mean'].min():.0f}-{df['wpd_mean'].max():.0f} W/m2")

    all_months = pd.concat(frames, ignore_index=True)

    annual = all_months.groupby(["lat", "lon"], as_index=False).agg(
        wpd_mean_annual  = ("wpd_mean",  "mean"),
        ws_mean_annual   = ("ws_mean",   "mean"),
        rho_mean_annual  = ("rho_mean",  "mean"),
        swh_p90_annual   = ("swh_p90",   "mean"),
        swh_max_annual   = ("swh_max",   "max"),
        mwp_mean_annual  = ("mwp_mean",  "mean"),
        dist_to_shore_km = ("dist_to_shore_km", "first"),
    )

    annual.to_parquet(ERA5_PQ, index=False)
    print(f"\n    Annual means -> {ERA5_PQ}")
    print(f"    Rows: {len(annual):,}  |  File: {ERA5_PQ.stat().st_size / 1e6:.1f} MB")
    print(f"    Annual WPD range: {annual['wpd_mean_annual'].min():.1f} - "
          f"{annual['wpd_mean_annual'].max():.1f} W/m2")

    del all_months, frames
    gc.collect()

    golden = spatial_join_golden_zones(ERA5_PQ, BATHY_PQ, RESULT_PQ)
    prime = compute_suitability_and_cluster(golden)
    plot_wpd_heatmap(annual, OUT_DIR / "wpd_heatmap.png")
    plot_golden_zones(golden, OUT_DIR / "golden_zones.png")
    print_top5_report(golden)

    print(f"\n[Done] All outputs written.")


# ------------------------------------------------------------------------------
# SECTION 5 — Spatial Join & Golden Zone filter via DuckDB
# ------------------------------------------------------------------------------

GOLDEN_ZONE_SQL = """
WITH era5 AS (
    SELECT
        lat,
        lon,
        wpd_mean_annual,
        ws_mean_annual,
        rho_mean_annual,
        swh_p90_annual,
        swh_max_annual,
        mwp_mean_annual,
        dist_to_shore_km,
        ROUND(lat / 0.25) * 0.25  AS lat_snap,
        ROUND(lon / 0.25) * 0.25  AS lon_snap
    FROM read_parquet('{era5_pq}')
    WHERE wpd_mean_annual >= {wpd_min}
      AND (swh_p90_annual  < {swh_p90_max} OR swh_p90_annual IS NULL)
      AND (swh_max_annual  < {swh_max_hard} OR swh_max_annual IS NULL)
),
bathy AS (
    SELECT
        ROUND(lat  / 0.25) * 0.25 AS lat_snap,
        ROUND(lon  / 0.25) * 0.25 AS lon_snap,
        AVG(ocean_depth_m)         AS depth_m,
        AVG(slope_deg)             AS slope_deg,
        BOOL_OR(slope_unstable)    AS slope_unstable,
        MIN(foundation_type)       AS foundation_type
    FROM read_parquet('{bathy_pq}')
    WHERE ocean_depth_m BETWEEN {d_min} AND {d_max}
      AND NOT land_mask
    GROUP BY lat_snap, lon_snap
),
joined AS (
    SELECT
        e.lat,
        e.lon,
        b.depth_m,
        b.slope_deg,
        b.slope_unstable,
        b.foundation_type,
        e.wpd_mean_annual,
        e.ws_mean_annual,
        e.rho_mean_annual,
        e.swh_p90_annual,
        e.swh_max_annual,
        e.mwp_mean_annual,
        e.dist_to_shore_km,
        (e.wpd_mean_annual / 1000.0)
            * (1.0 - LEAST(b.slope_deg / 30.0, 1.0))
            * (1.0 - LEAST(COALESCE(e.swh_p90_annual, 0) / 8.0, 1.0))
            AS composite_score
    FROM era5 e
    INNER JOIN bathy b
        ON e.lat_snap = b.lat_snap
       AND e.lon_snap = b.lon_snap
    WHERE NOT b.slope_unstable
)
SELECT * FROM joined
ORDER BY composite_score DESC
"""


def spatial_join_golden_zones(
    era5_pq: Path,
    bathy_pq: Path,
    result_pq: Path,
) -> pd.DataFrame:
    print(f"\n[3] DuckDB spatial join - filtering Golden Zones ...")
    print(f"    Filters: WPD >= {WPD_MIN_W_M2} W/m2, "
          f"depth {DEPTH_MIN_M}-{DEPTH_MAX_M} m, "
          f"SWH p90 < {SWH_P90_MAX} m, SWH max < {SWH_MAX_HARD} m, slope stable")

    sql = GOLDEN_ZONE_SQL.format(
        era5_pq=str(era5_pq),
        bathy_pq=str(bathy_pq),
        wpd_min=WPD_MIN_W_M2,
        swh_p90_max=SWH_P90_MAX,
        swh_max_hard=SWH_MAX_HARD,
        d_min=DEPTH_MIN_M,
        d_max=DEPTH_MAX_M,
    )

    con = duckdb.connect()
    df = con.execute(sql).df()
    con.close()

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.to_parquet(result_pq, index=False)
    print(f"    Golden Zone cells found: {len(df):,}")
    print(f"    Saved -> {result_pq}")
    return df


# ------------------------------------------------------------------------------
# SECTION 6 — Normalized Suitability Score + DBSCAN Clustering
# ------------------------------------------------------------------------------

def compute_suitability_and_cluster(golden_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[4] Computing Normalized Suitability Score ...")

    if golden_df.empty:
        print("    No data to score.")
        return golden_df

    df = golden_df.copy()

    scaler = MinMaxScaler()
    wpd_norm = scaler.fit_transform(df[["wpd_mean_annual"]])[:, 0]

    wave_norm = MinMaxScaler().fit_transform(df[["swh_max_annual"]])[:, 0]
    wave_calm = 1.0 - wave_norm

    depth_vals = df["depth_m"].values
    ideal_depth = np.where(depth_vals <= 60, 37.5, 180.0)
    max_deviation = np.where(depth_vals <= 60, 22.5, 120.0)
    depth_factor = 1.0 - np.clip(
        np.abs(depth_vals - ideal_depth) / max_deviation, 0, 1
    )

    df["suitability"] = W_WPD * wpd_norm + W_DEPTH * depth_factor + W_WAVE * wave_calm

    print(f"    Suitability range: {df['suitability'].min():.3f} - "
          f"{df['suitability'].max():.3f}")

    threshold = df["suitability"].quantile(0.90)
    top10 = df[df["suitability"] >= threshold].copy()
    print(f"    Top 10% threshold: {threshold:.3f}  ({len(top10):,} cells)")

    print(f"    DBSCAN eps={DBSCAN_EPS} deg, min_samples={DBSCAN_MIN_PTS} ...")
    coords = top10[["lat", "lon"]].values
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS).fit(coords)
    top10["cluster_id"] = clustering.labels_

    n_clusters = len(set(clustering.labels_) - {-1})
    n_noise    = (clustering.labels_ == -1).sum()
    print(f"    Clusters found: {n_clusters}  |  Noise points: {n_noise}")

    df["cluster_id"] = -1
    df.loc[top10.index, "cluster_id"] = top10["cluster_id"].values

    df = df.sort_values("suitability", ascending=False).reset_index(drop=True)
    df.to_parquet(PRIME_PQ, index=False)
    print(f"    Saved -> {PRIME_PQ}")

    if n_clusters > 0:
        cluster_cells = top10[top10["cluster_id"] >= 0]
        cluster_sizes = (
            cluster_cells.groupby("cluster_id")
            .size()
            .sort_values(ascending=False)
        )
        top3_ids = cluster_sizes.head(3).index.tolist()

        print(f"\n    === TOP {min(3, n_clusters)} CLUSTER CENTERS ===")
        for rank, cid in enumerate(top3_ids, 1):
            members = cluster_cells[cluster_cells["cluster_id"] == cid]
            center_lat = members["lat"].mean()
            center_lon = members["lon"].mean()
            mean_suit  = members["suitability"].mean()
            print(f"    Cluster {rank} (id={cid}): "
                  f"center=({center_lat:.2f}N, {center_lon:.2f}E), "
                  f"cells={len(members)}, "
                  f"mean_suitability={mean_suit:.3f}")

    return df


# ------------------------------------------------------------------------------
# SECTION 7 — Visualisations
# ------------------------------------------------------------------------------

def plot_wpd_heatmap(annual_df: pd.DataFrame, out_path: Path) -> None:
    print("\n[5a] Plotting WPD heatmap ...")
    lats = np.sort(annual_df["lat"].unique())
    lons = np.sort(annual_df["lon"].unique())

    wpd_pivot = annual_df.pivot_table(
        values="wpd_mean_annual", index="lat", columns="lon",
    ).reindex(index=lats, columns=lons)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    cmap = cmocean.cm.matter if HAS_CMOCEAN else plt.cm.YlOrRd
    cmap.set_bad("#d0d0d0")
    vmax = float(np.nanpercentile(wpd_pivot.values, 99))

    img = ax.pcolormesh(lons, lats, wpd_pivot.values,
                        cmap=cmap, vmin=0, vmax=vmax,
                        shading="auto", rasterized=True)
    cbar = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean WPD (W/m2)", fontsize=11)
    ax.set_title(f"ERA5 {YEAR} - Annual Mean Wind Power Density (100 m)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {out_path}")


def plot_golden_zones(gdf: pd.DataFrame, out_path: Path) -> None:
    print("\n[5b] Plotting Golden Zone scatter ...")
    if gdf.empty:
        print("    No Golden Zone cells - skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    sc = ax.scatter(gdf["lon"], gdf["lat"],
                    c=gdf["wpd_mean_annual"], cmap="YlOrRd",
                    s=8, alpha=0.7, edgecolors="none",
                    vmin=WPD_MIN_W_M2,
                    vmax=float(gdf["wpd_mean_annual"].quantile(0.99)))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean WPD (W/m2)", fontsize=10)

    top5 = gdf.head(5)
    for i, row in top5.iterrows():
        ax.annotate(f"#{i+1}\n{row['lat']:.2f}N\n{row['lon']:.2f}",
                    xy=(row["lon"], row["lat"]),
                    xytext=(row["lon"] + 1, row["lat"] + 0.5),
                    fontsize=7, color="navy",
                    arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))

    ax.set_title(f"ERA5 {YEAR} - Offshore Wind Golden Zones\n"
                 f"(WPD≥{WPD_MIN_W_M2} W/m², depth {DEPTH_MIN_M}-{DEPTH_MAX_M} m, "
                 f"SWH p90<{SWH_P90_MAX} m)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ------------------------------------------------------------------------------
# SECTION 8 — Top 5 Summary Report
# ------------------------------------------------------------------------------

FOUNDATION_LABELS = {
    0: "Monopile (0-15 m)",
    1: "Jacket/Fixed (15-60 m)",
    2: "Floating (60-300 m)",
    3: "Infeasible (>300 m)",
}


def print_top5_report(gdf: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"  TOP 5 GOLDEN ZONE SITES - ERA5 {YEAR} + GEBCO 2025")
    print("=" * 70)

    if gdf.empty:
        print("  No sites passed all Golden Zone filters.")
        return

    cols = ["lat", "lon", "depth_m", "wpd_mean_annual",
            "ws_mean_annual", "swh_max_annual", "slope_deg",
            "foundation_type", "composite_score"]

    top5 = gdf[cols].head(5).reset_index(drop=True)
    top5["foundation_type"] = top5["foundation_type"].map(
        lambda x: FOUNDATION_LABELS.get(int(x), "Unknown") if pd.notna(x) else "Unknown"
    )

    for rank, row in top5.iterrows():
        energy_risk = "HIGH" if row["wpd_mean_annual"] > 600 else "MODERATE"
        eng_risk = ("LOW" if not gdf.iloc[rank]["slope_unstable"]
                          and row["swh_max_annual"] < 1.5
                    else "MODERATE")
        print(f"\n  #{rank + 1}  ({row['lat']:.3f}N, {row['lon']:.3f})")
        print(f"       Foundation    : {row['foundation_type']}")
        print(f"       Depth         : {row['depth_m']:.1f} m")
        print(f"       Mean WPD      : {row['wpd_mean_annual']:.1f} W/m2")
        print(f"       Mean Wind     : {row['ws_mean_annual']:.2f} m/s")
        print(f"       Max Wave Ht   : {row['swh_max_annual']:.2f} m")
        print(f"       Slope         : {row['slope_deg']:.2f} deg")
        print(f"       Composite     : {row['composite_score']:.4f}")
        print(f"       Energy Pot.   : {energy_risk} | Eng. Risk: {eng_risk}")

    print("\n" + "=" * 70)
    print(f"  Total qualifying cells: {len(gdf):,}")
    print(f"  Full results -> {RESULT_PQ}")
    print("=" * 70)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python {sys.argv[0]} month <1-12>    Process a single month")
        print(f"  python {sys.argv[0]} aggregate        Combine monthly parquets -> suitability")
        print()
        print()
        print("Example workflow:")
        print(f"  for m in 1 2 3 4 5 6 7 8 9 10 11 12; do")
        print(f"    python {sys.argv[0]} month $m")
        print("  done")
        print(f"  python {sys.argv[0]} aggregate")
        sys.exit(1)

    mode = sys.argv[1].lower()

    try:
        if mode == "month":
            if len(sys.argv) < 3:
                print("ERROR: specify month number (1-12)")
                sys.exit(1)
            month = int(sys.argv[2])
            if not 1 <= month <= 12:
                print("ERROR: month must be 1-12")
                sys.exit(1)
            run_single_month(month)

        elif mode == "aggregate":
            run_aggregate()

        else:
            print(f"Unknown mode: '{mode}'. Use 'month' or 'aggregate'.")
            sys.exit(1)

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
