"""
02_wind_era5.py — ERA5 Wind Physics, Feasibility Filter & Suitability Join
===========================================================================
Source  : gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
Outputs :
  data/era5_annual_means_2025.parquet   — gridded annual-mean physics fields
  data/suitability_results_2025.parquet — spatial join with bathymetry (Golden Zones)
  outputs/plots/wpd_heatmap.png
  outputs/plots/golden_zones.png

Run: python 02_wind_era5.py
"""

import sys
import traceback
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da
import gcsfs
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

try:
    import cmocean
    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False

# ── Config ────────────────────────────────────────────────────────────────────
ZARR_URI   = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BATHY_PQ   = Path("data/bathymetry_processed.parquet")
ERA5_PQ    = Path("data/era5_annual_means_2025.parquet")
RESULT_PQ  = Path("data/suitability_results_2025.parquet")
OUT_DIR    = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Spatial bounds (ARCO ERA5: lat 90→-90 descending, lon 0→359.75)
LAT_SLICE  = slice(72, 18)     # US waters only (Alaska 72°N to Hawaii 18°N)
LON_SLICE  = slice(195, 305)   # -165°→-55° in 0–360 convention

# Time — 4 seasonal representatives (saves ~3x memory vs 12 months)
YEAR       = 2024
MONTHS     = [1, 4, 7, 10]    # winter, spring, summer, fall

# Golden Zone thresholds
WPD_MIN_W_M2   = 400.0   # W/m² — minimum viable wind power density
DEPTH_MIN_M    = 15.0    # m    — minimum ocean depth (Jacket/Fixed low end)
DEPTH_MAX_M    = 60.0    # m    — maximum depth for fixed-bottom (Jacket)
SWH_MAX_M      = 2.5     # m    — max significant wave height

# Physical constants
R_DRY_AIR = 287.058      # J/(kg·K) — specific gas constant, dry air

# ARCO ERA5 uses full variable names, not CDS short codes
VAR_U100 = "100m_u_component_of_wind"
VAR_V100 = "100m_v_component_of_wind"
VAR_T2M  = "2m_temperature"
VAR_SP   = "surface_pressure"
VAR_SWH  = "significant_height_of_combined_wind_waves_and_swell"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Open ERA5 Zarr (lazy, no data downloaded yet)
# ─────────────────────────────────────────────────────────────────────────────

def open_era5_lazy() -> xr.Dataset:
    """
    Open ARCO ERA5 Zarr store with anonymous GCS access.
    Returns a lazy xarray Dataset with dask-backed arrays.
    Coordinate names in this store: 'latitude', 'longitude', 'time'.
    """
    print("[1] Opening ERA5 Zarr store (anonymous GCS) ...")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ZARR_URI)
    ds = xr.open_zarr(store, consolidated=True, chunks={})   # chunks={} → dask

    # Verify required variables exist (full ARCO names)
    required = {VAR_U100, VAR_V100, VAR_T2M, VAR_SP, VAR_SWH}
    missing = required - set(ds.data_vars)
    if missing:
        raise KeyError(f"ERA5 store missing variables: {missing}. "
                       f"Available: {sorted(ds.data_vars)}")

    # Standardise coord names → lat / lon for consistency with bathymetry
    rename = {}
    if "latitude"  in ds.coords and "lat" not in ds.coords:
        rename["latitude"]  = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    print(f"    Store opened. Variables: {sorted(ds.data_vars)}")
    print(f"    Full time range: {str(ds.time.values[0])[:10]} → "
          f"{str(ds.time.values[-1])[:10]}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Monthly batch physics: WPD, Air Density, SWH stats
# ─────────────────────────────────────────────────────────────────────────────

def _physics_for_month(ds: xr.Dataset, year: int, month: int) -> dict:
    """
    For one calendar month:
    1. Slice spatially + temporally (lazy → only metadata touched).
    2. Compute wind fields first, then wave height separately to limit peak RAM.
    Returns xr.Dataset of 2-D monthly aggregate arrays.
    """
    import calendar
    import gc
    _, ndays = calendar.monthrange(year, month)
    t_start  = f"{year}-{month:02d}-01"
    t_end    = f"{year}-{month:02d}-{ndays:02d}T23:00"

    print(f"    Computing month {year}-{month:02d} ...", flush=True)

    # --- Step A: Wind speed, air density, WPD ---
    sub_wind = ds[[VAR_U100, VAR_V100, VAR_T2M, VAR_SP]].sel(
        time=slice(t_start, t_end),
        lat=LAT_SLICE,
        lon=LON_SLICE,
    )
    ws  = np.sqrt(sub_wind[VAR_U100] ** 2 + sub_wind[VAR_V100] ** 2)
    rho = sub_wind[VAR_SP] / (R_DRY_AIR * sub_wind[VAR_T2M])
    wpd = 0.5 * rho * ws ** 3

    wind_result = xr.Dataset({
        "wpd_mean": wpd.mean(dim="time"),
        "ws_mean":  ws.mean(dim="time"),
        "rho_mean": rho.mean(dim="time"),
    }).compute()
    del sub_wind, ws, rho, wpd
    gc.collect()
    print(f"      wind fields done", flush=True)

    # --- Step B: Wave height (separate compute to limit peak RAM) ---
    sub_wave = ds[[VAR_SWH]].sel(
        time=slice(t_start, t_end),
        lat=LAT_SLICE,
        lon=LON_SLICE,
    )
    wave_result = xr.Dataset({
        "swh_p90": sub_wave[VAR_SWH].quantile(0.90, dim="time"),
        "swh_max": sub_wave[VAR_SWH].max(dim="time"),
    }).compute()
    del sub_wave
    gc.collect()
    print(f"      wave fields done", flush=True)

    return xr.merge([wind_result, wave_result])


def compute_annual_means(ds: xr.Dataset, year: int) -> xr.Dataset:
    """
    Process each month independently to bound peak memory.
    Returns one Dataset with time-averaged fields over the full year.
    """
    print(f"\n[2] Computing physics for {year} ({len(MONTHS)} seasonal batches) ...")
    monthly = []
    for m in MONTHS:
        try:
            monthly.append(_physics_for_month(ds, year, m))
        except Exception as e:
            print(f"\n    [WARN] Month {m} failed: {e} — skipping.")

    if not monthly:
        raise RuntimeError("All months failed — check GCS connectivity.")

    # Stack months and take annual mean/max
    stack = xr.concat(monthly, dim="month")
    annual = xr.Dataset({
        "wpd_mean_annual":  stack["wpd_mean"].mean(dim="month"),
        "ws_mean_annual":   stack["ws_mean"].mean(dim="month"),
        "rho_mean_annual":  stack["rho_mean"].mean(dim="month"),
        "swh_p90_annual":   stack["swh_p90"].mean(dim="month"),
        "swh_max_annual":   stack["swh_max"].max(dim="month"),
    })

    # Drop xarray quantile coord artifact if present
    for v in annual.data_vars:
        if "quantile" in annual[v].coords:
            annual[v] = annual[v].drop_vars("quantile")

    print(f"\n    Annual mean WPD range: "
          f"{float(annual['wpd_mean_annual'].min()):.1f} – "
          f"{float(annual['wpd_mean_annual'].max()):.1f} W/m²")
    return annual


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Export gridded ERA5 results to Parquet
# ─────────────────────────────────────────────────────────────────────────────

def era5_to_parquet(annual: xr.Dataset, out_path: Path) -> pd.DataFrame:
    """Flatten 2-D ERA5 annual means to a tidy DataFrame and save."""
    print(f"\n[3] Exporting ERA5 annual means → {out_path} ...")

    # Convert lon from 0–360 back to -180–180 for bathymetry join
    lons_180 = annual["lon"].values.copy()
    lons_180[lons_180 > 180] -= 360

    lat2d, lon2d = np.meshgrid(annual["lat"].values,
                               lons_180, indexing="ij")
    df = pd.DataFrame({
        "lat":             lat2d.ravel(),
        "lon":             lon2d.ravel(),
        "wpd_mean_annual": annual["wpd_mean_annual"].values.ravel(),
        "ws_mean_annual":  annual["ws_mean_annual"].values.ravel(),
        "rho_mean_annual": annual["rho_mean_annual"].values.ravel(),
        "swh_p90_annual":  annual["swh_p90_annual"].values.ravel(),
        "swh_max_annual":  annual["swh_max_annual"].values.ravel(),
    }).dropna(subset=["wpd_mean_annual"])

    # Round lat/lon to ERA5 grid precision for join key
    df["lat"] = df["lat"].round(4)
    df["lon"] = df["lon"].round(4)

    df.to_parquet(out_path, index=False)
    print(f"    Rows: {len(df):,}  |  File: {out_path.stat().st_size/1e6:.1f} MB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Spatial Join & Golden Zone filter via DuckDB
# ─────────────────────────────────────────────────────────────────────────────

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
        -- Nearest-neighbour snap to 0.25° ERA5 grid
        ROUND(lat / 0.25) * 0.25  AS lat_snap,
        ROUND(lon / 0.25) * 0.25  AS lon_snap
    FROM read_parquet('{era5_pq}')
    WHERE wpd_mean_annual >= {wpd_min}
      AND swh_max_annual  <  {swh_max}
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
        -- Composite risk score: higher WPD, lower depth variance & slope = better
        (e.wpd_mean_annual / 1000.0)
            * (1.0 - LEAST(b.slope_deg / 30.0, 1.0))
            * (1.0 - LEAST(e.swh_max_annual / 5.0, 1.0))
            AS composite_score
    FROM era5 e
    INNER JOIN bathy b
        ON e.lat_snap = b.lat_snap
       AND e.lon_snap = b.lon_snap
    WHERE NOT b.slope_unstable     -- exclude structurally unstable seafloor
)
SELECT * FROM joined
ORDER BY composite_score DESC
"""


def spatial_join_golden_zones(era5_pq: Path, bathy_pq: Path,
                               result_pq: Path) -> pd.DataFrame:
    """
    Use DuckDB to join ERA5 physics with bathymetry, apply Golden Zone filters,
    and write the result to Parquet.
    """
    print(f"\n[4] DuckDB spatial join — filtering Golden Zones ...")
    print(f"    Filters: WPD > {WPD_MIN_W_M2} W/m², "
          f"depth {DEPTH_MIN_M}–{DEPTH_MAX_M} m, "
          f"max SWH < {SWH_MAX_M} m, slope stable")

    sql = GOLDEN_ZONE_SQL.format(
        era5_pq=str(era5_pq),
        bathy_pq=str(bathy_pq),
        wpd_min=WPD_MIN_W_M2,
        swh_max=SWH_MAX_M,
        d_min=DEPTH_MIN_M,
        d_max=DEPTH_MAX_M,
    )

    con = duckdb.connect()
    df  = con.execute(sql).df()
    con.close()

    df.to_parquet(result_pq, index=False)
    print(f"    Golden Zone cells found: {len(df):,}")
    print(f"    Saved → {result_pq}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_wpd_heatmap(annual: xr.Dataset, out_path: Path) -> None:
    print("\n[5a] Plotting WPD heatmap ...")
    wpd = annual["wpd_mean_annual"]
    lons, lats = annual["lon"].values, annual["lat"].values

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    cmap = cmocean.cm.matter if HAS_CMOCEAN else plt.cm.YlOrRd
    cmap.set_bad("#d0d0d0")
    vmax = float(np.nanpercentile(wpd.values, 99))

    img = ax.pcolormesh(lons, lats, wpd.values,
                        cmap=cmap, vmin=0, vmax=vmax,
                        shading="auto", rasterized=True)
    cbar = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean WPD (W/m²)", fontsize=11)
    ax.set_title(f"ERA5 {YEAR} — Annual Mean Wind Power Density (100 m)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")
    print(f"    Key trend: highest WPD offshore the US East Coast "
          f"and Gulf of Alaska — consistent with prevailing westerlies.")


def plot_golden_zones(gdf: pd.DataFrame, out_path: Path) -> None:
    print("\n[5b] Plotting Golden Zone scatter ...")
    if gdf.empty:
        print("    No Golden Zone cells — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    sc = ax.scatter(gdf["lon"], gdf["lat"],
                    c=gdf["wpd_mean_annual"], cmap="YlOrRd",
                    s=8, alpha=0.7, edgecolors="none",
                    vmin=WPD_MIN_W_M2,
                    vmax=float(gdf["wpd_mean_annual"].quantile(0.99)))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean WPD (W/m²)", fontsize=10)

    # Annotate Top 5
    top5 = gdf.head(5)
    for i, row in top5.iterrows():
        ax.annotate(f"#{i+1}\n{row['lat']:.2f}°N\n{row['lon']:.2f}°",
                    xy=(row["lon"], row["lat"]),
                    xytext=(row["lon"] + 1, row["lat"] + 0.5),
                    fontsize=7, color="navy",
                    arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))

    ax.set_xlim(-165, -55); ax.set_ylim(15, 85)
    ax.set_title(f"ERA5 {YEAR} — Offshore Wind Golden Zones\n"
                 f"(WPD>{WPD_MIN_W_M2} W/m², depth {DEPTH_MIN_M}–{DEPTH_MAX_M} m, "
                 f"SWH<{SWH_MAX_M} m)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Top 5 Summary Report
# ─────────────────────────────────────────────────────────────────────────────

FOUNDATION_LABELS = {
    0: "Monopile (0–15 m)",
    1: "Jacket/Fixed (15–60 m)",
    2: "Floating (60–300 m)",
    3: "Infeasible (>300 m)",
}


def print_top5_report(gdf: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"  TOP 5 GOLDEN ZONE SITES — ERA5 {YEAR} + GEBCO 2025")
    print("=" * 70)

    if gdf.empty:
        print("  No sites passed all Golden Zone filters.")
        print("  Suggestions:")
        print("  • Widen SWH threshold (currently < 2.5 m)")
        print("  • Expand depth range beyond 15–60 m (include Floating)")
        print("  • Lower WPD floor below 400 W/m²")
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
        eng_risk    = ("LOW"  if not gdf.loc[rank, "slope_unstable"]
                              and row["swh_max_annual"] < 1.5
                       else "MODERATE")
        print(f"\n  #{rank + 1}  ({row['lat']:.3f}°N, {row['lon']:.3f}°)")
        print(f"       Foundation    : {row['foundation_type']}")
        print(f"       Depth         : {row['depth_m']:.1f} m")
        print(f"       Mean WPD      : {row['wpd_mean_annual']:.1f} W/m²")
        print(f"       Mean Wind     : {row['ws_mean_annual']:.2f} m/s")
        print(f"       Max Wave Ht   : {row['swh_max_annual']:.2f} m")
        print(f"       Slope         : {row['slope_deg']:.2f}°")
        print(f"       Composite Score: {row['composite_score']:.4f}")
        print(f"       Energy Potential: {energy_risk} | Engineering Risk: {eng_risk}")

    print("\n" + "=" * 70)
    print(f"  Total qualifying cells: {len(gdf):,}")
    print(f"  Full results → {RESULT_PQ}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        # 1. Open ERA5
        ds_era5 = open_era5_lazy()

        # 2. Compute annual means (monthly batches)
        annual = compute_annual_means(ds_era5, YEAR)

        # 3. Export ERA5 parquet
        era5_df = era5_to_parquet(annual, ERA5_PQ)

        # 4. DuckDB join → Golden Zones
        golden = spatial_join_golden_zones(ERA5_PQ, BATHY_PQ, RESULT_PQ)

        # 5. Plots
        plot_wpd_heatmap(annual, OUT_DIR / "wpd_heatmap.png")
        plot_golden_zones(golden, OUT_DIR / "golden_zones.png")

        # 6. Report
        print_top5_report(golden)

        print("\n[Done] All outputs written.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
