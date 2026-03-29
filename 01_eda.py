"""
01_eda.py — Bathymetry EDA & Preprocessing for Offshore Wind Suitability
=========================================================================
Data source : GEBCO 2025 NetCDF (gebco_2025_n80.0_s20.0_w-160.0_e-60.0.nc)
Outputs     : ./outputs/plots/  (depth_heatmap.png, foundation_histogram.png)
              ./data/           (bathymetry_processed.parquet  — intermediary)

Run:  python 01_eda.py
Notebook:  Each section maps 1-to-1 with a Jupyter cell block.
"""

import sys
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    import cmocean
    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False
    print("[WARN] cmocean not installed — falling back to 'viridis_r'.")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
NC_FILE    = DATA_DIR / "gebco_2025_n80.0_s20.0_w-160.0_e-60.0.nc"
OUT_DIR    = Path("outputs") / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Subsample stride for heavy operations (set to 1 for full resolution, costs RAM/time)
STRIDE = 10   # 1/10 resolution ≈ ~4km grid — sufficient for EDA

RANDOM_STATE = 42

# ── US Coastal Zone Polygons (approximate EEZ boundaries) ────────────────────
# Each polygon encloses US coastal waters (~200 nm offshore) while excluding
# neighbouring countries (Mexico, Canada, Cuba, Bahamas).
# Coordinates are (lon, lat) per Shapely convention.

US_EAST_COAST = Polygon([
    (-81.5, 24.5),   # south Florida
    (-79.5, 24.5),   # Florida Straits (exclude Cuba)
    (-75.0, 26.0),   # off Bahamas
    (-64.0, 30.0),   # mid-Atlantic offshore
    (-63.0, 40.0),   # off New England
    (-65.0, 45.0),   # Gulf of Maine
    (-66.5, 45.0),   # Maine / Canada border
    (-67.0, 47.5),   # northern Maine
    (-68.0, 47.5),
    (-71.0, 45.0),   # inland boundary (coast side)
    (-76.0, 39.0),   # Chesapeake
    (-78.0, 34.0),   # Carolinas coast
    (-80.5, 31.0),   # Georgia coast
    (-81.5, 25.5),   # Florida east coast
    (-81.5, 24.5),
])

US_GULF_COAST = Polygon([
    (-81.5, 24.5),   # south Florida
    (-81.5, 25.5),
    (-82.5, 28.5),   # Florida west coast
    (-84.0, 30.0),   # Florida panhandle
    (-88.0, 30.5),   # Alabama/Mississippi
    (-90.5, 29.5),   # Louisiana
    (-94.0, 29.5),   # Texas coast
    (-97.5, 26.0),   # south Texas / Mexico border
    (-97.5, 24.5),   # offshore Texas
    (-93.0, 25.0),   # central Gulf (US side)
    (-86.0, 24.0),   # exclude Cuban waters
    (-81.5, 24.5),
])

US_WEST_COAST = Polygon([
    (-117.2, 32.5),  # San Diego / Mexico border
    (-122.0, 32.5),  # offshore southern CA
    (-128.0, 38.0),  # offshore central CA
    (-130.0, 43.0),  # offshore Oregon
    (-130.0, 49.0),  # offshore Washington
    (-124.8, 49.0),  # US / Canada border (48.5°N at coast)
    (-124.5, 48.5),
    (-123.5, 46.5),  # Washington coast
    (-124.5, 43.0),  # Oregon coast
    (-124.0, 40.0),  # northern CA
    (-121.0, 35.0),  # central CA
    (-117.5, 33.0),  # southern CA
    (-117.2, 32.5),
])

US_ALASKA = Polygon([
    (-160.0, 52.0),  # Aleutians (western limit of data)
    (-160.0, 72.0),  # north slope
    (-141.0, 72.0),  # US / Canada border
    (-141.0, 60.0),  # SE Alaska / Canada border
    (-135.0, 55.0),  # SE Alaska panhandle
    (-133.0, 54.5),  # Dixon Entrance
    (-130.0, 52.0),  # offshore SE Alaska
    (-160.0, 52.0),
])

US_HAWAII = Polygon([
    (-162.0, 18.5),  # NW extent
    (-162.0, 23.5),
    (-154.0, 23.5),
    (-154.0, 18.5),
    (-162.0, 18.5),
])

US_PUERTO_RICO_USVI = Polygon([
    (-68.0, 17.0),
    (-68.0, 19.0),
    (-64.0, 19.0),
    (-64.0, 17.0),
    (-68.0, 17.0),
])

US_COASTAL_ZONES = unary_union([
    US_EAST_COAST, US_GULF_COAST, US_WEST_COAST,
    US_ALASKA, US_HAWAII, US_PUERTO_RICO_USVI,
])
US_COASTAL_ZONES_PREP = prep(US_COASTAL_ZONES)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_bathymetry(nc_path: Path) -> xr.Dataset:
    """Load GEBCO NetCDF and standardise coordinate names to 'lat'/'lon'."""
    print(f"[1] Loading {nc_path} ...")
    ds = xr.open_dataset(nc_path, engine="netcdf4")

    # Normalise coordinate names
    rename_map = {}
    for name in ds.coords:
        lower = name.lower()
        if lower in ("latitude", "y") and "lat" not in ds.coords:
            rename_map[name] = "lat"
        if lower in ("longitude", "x") and "lon" not in ds.coords:
            rename_map[name] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    print(f"    Dimensions : {dict(ds.dims)}")
    print(f"    Variables  : {list(ds.data_vars)}")
    print(f"    Lat range  : {float(ds.lat.min()):.2f} → {float(ds.lat.max()):.2f}")
    print(f"    Lon range  : {float(ds.lon.min()):.2f} → {float(ds.lon.max()):.2f}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(ds: xr.Dataset, stride: int = STRIDE) -> xr.Dataset:
    """
    Subsample, convert depth sign, mask land.
    Returns a lightweight xarray Dataset ready for analysis.
    """
    print(f"\n[2] Preprocessing (stride={stride}) ...")

    # Subsample for EDA speed
    ds_sub = ds.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))

    elev = ds_sub["elevation"].load()   # load into memory after slicing

    # Depth = positive below sea surface (negate GEBCO elevation)
    depth = -elev                        # land cells become negative
    depth.name = "depth_m"
    depth.attrs = {"units": "m", "positive": "down",
                   "long_name": "Water depth (positive below sea surface)"}

    # Land mask: True where elevation > 0 (land)
    land_mask = elev > 0
    land_mask.name = "land_mask"

    # Ocean depth (NaN over land)
    ocean_depth = depth.where(~land_mask)
    ocean_depth.name = "ocean_depth_m"

    result = xr.Dataset(
        {"elevation": elev, "depth_m": depth,
         "land_mask": land_mask, "ocean_depth_m": ocean_depth},
        coords=ds_sub.coords,
    )

    print(f"    Subsampled shape : {ocean_depth.shape}")
    print(f"    Ocean depth range: {float(ocean_depth.min()):.1f} m "
          f"→ {float(ocean_depth.max()):.1f} m")
    print(f"    Land fraction    : {float(land_mask.mean())*100:.1f} %")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Foundation Classification
# ─────────────────────────────────────────────────────────────────────────────

FOUNDATION_BINS   = [0, 15, 60, 300, np.inf]
FOUNDATION_LABELS = ["Monopile (0–15 m)",
                     "Jacket/Fixed (15–60 m)",
                     "Floating (60–300 m)",
                     "Infeasible (>300 m)"]
FOUNDATION_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]


def classify_foundation(ds: xr.Dataset) -> xr.Dataset:
    """
    Add 'foundation_type' (int 0–3) and 'foundation_label' (string) arrays.
    Ocean cells only; land = NaN / empty.
    """
    print("\n[3] Classifying foundation types ...")

    depth = ds["ocean_depth_m"].values.copy()    # NaN over land

    ftype = np.full(depth.shape, np.nan)
    for i, (lo, hi) in enumerate(zip(FOUNDATION_BINS[:-1], FOUNDATION_BINS[1:])):
        mask = (depth >= lo) & (depth < hi)
        ftype[mask] = i

    foundation = xr.DataArray(
        ftype, coords=ds["ocean_depth_m"].coords, name="foundation_type",
        attrs={"flag_values": list(range(len(FOUNDATION_LABELS))),
               "flag_meanings": " ".join(FOUNDATION_LABELS),
               "long_name": "Foundation type index (0=Monopile … 3=Infeasible)"},
    )
    ds = ds.assign(foundation_type=foundation)

    for i, label in enumerate(FOUNDATION_LABELS):
        count = int(np.nansum(ftype == i))
        print(f"    [{i}] {label:30s}: {count:,} cells")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Slope Analysis
# ─────────────────────────────────────────────────────────────────────────────

SLOPE_THRESHOLD_DEG = 5.0


def compute_slope(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute seafloor slope (degrees) from 2-D gradient of ocean depth.
    Flags cells with slope > SLOPE_THRESHOLD_DEG as structurally unstable.

    Gradient is computed in metres/metre (unit-consistent) using the actual
    lat/lon grid spacing converted to metres at each latitude.
    """
    print(f"\n[4] Computing slope (unstable threshold = {SLOPE_THRESHOLD_DEG}°) ...")

    depth = ds["ocean_depth_m"].values          # (lat, lon), NaN over land
    lats  = ds["lat"].values
    lons  = ds["lon"].values

    # Grid spacing in metres
    d_lat_deg = abs(float(lats[1] - lats[0]))
    d_lon_deg = abs(float(lons[1] - lons[0]))
    d_lat_m   = d_lat_deg * 111_320.0           # ~111.32 km per degree latitude

    # Gradient arrays (forward difference, NaN-safe)
    dy, dx = np.gradient(depth)                  # in depth-units per cell

    # Convert cell counts to metres for each row (longitude spacing varies with lat)
    cos_lat = np.cos(np.radians(lats))[:, np.newaxis]
    d_lon_m = d_lon_deg * 111_320.0 * cos_lat

    slope_frac = np.sqrt((dy / d_lat_m) ** 2 + (dx / d_lon_m) ** 2)
    slope_deg  = np.degrees(np.arctan(slope_frac))

    # Mask land cells
    ocean_mask = ~ds["land_mask"].values
    slope_deg[~ocean_mask] = np.nan

    slope_da = xr.DataArray(
        slope_deg, coords=ds["ocean_depth_m"].coords, name="slope_deg",
        attrs={"units": "degrees", "long_name": "Seafloor slope"},
    )
    unstable = xr.DataArray(
        slope_deg > SLOPE_THRESHOLD_DEG,
        coords=ds["ocean_depth_m"].coords, name="slope_unstable",
        attrs={"long_name": f"Slope > {SLOPE_THRESHOLD_DEG}° (structurally unstable)"},
    )

    ds = ds.assign(slope_deg=slope_da, slope_unstable=unstable)

    pct_unstable = float(np.nanmean(slope_deg > SLOPE_THRESHOLD_DEG)) * 100
    print(f"    Ocean cells with slope > {SLOPE_THRESHOLD_DEG}°: {pct_unstable:.2f} %")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4b — US Coastal Zone Filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_us_coastal(ds: xr.Dataset) -> xr.Dataset:
    """
    Mask out all grid cells that do NOT fall within US coastal zone polygons.
    Sets ocean variables to NaN and land_mask to True for non-US cells.
    """
    print("\n[4b] Filtering to US coastal zones ...")

    lats = ds["lat"].values
    lons = ds["lon"].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Vectorised point-in-polygon using shapely
    flat_lons = lon2d.ravel()
    flat_lats = lat2d.ravel()

    from shapely import contains_xy
    us_mask_flat = contains_xy(US_COASTAL_ZONES, flat_lons, flat_lats)
    us_mask = us_mask_flat.reshape(lon2d.shape)

    # Also require ocean (not land)
    ocean_mask = ~ds["land_mask"].values
    valid = us_mask & ocean_mask

    # Apply mask: set non-US-ocean cells to NaN
    for var in ["ocean_depth_m", "foundation_type", "slope_deg"]:
        if var in ds:
            ds[var] = ds[var].where(valid)

    ds["slope_unstable"] = ds["slope_unstable"].where(valid, other=False)

    # Update land_mask so non-US ocean cells are treated as masked
    new_land_mask = ~valid
    ds["land_mask"] = xr.DataArray(
        new_land_mask, coords=ds["land_mask"].coords, name="land_mask",
    )

    us_ocean_cells = int(valid.sum())
    print(f"    US ocean cells retained: {us_ocean_cells:,}")
    print(f"    Cells masked out       : {int((~valid).sum()):,}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EDA Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _depth_cmap():
    if HAS_CMOCEAN:
        return cmocean.cm.deep
    return plt.cm.viridis_r


def plot_depth_heatmap(ds: xr.Dataset, out_path: Path) -> None:
    """High-quality heatmap of ocean depth coloured by cmocean 'deep'."""
    print("\n[5a] Plotting depth heatmap ...")

    ocean_depth = ds["ocean_depth_m"]
    lons = ds["lon"].values
    lats = ds["lat"].values

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

    # Mask land as white/grey
    data = ocean_depth.values
    cmap = _depth_cmap()
    cmap.set_bad(color="#c8c8c8")   # land = mid-grey

    vmax = float(np.nanpercentile(data, 99))
    img = ax.pcolormesh(lons, lats, data,
                        cmap=cmap, vmin=0, vmax=vmax,
                        shading="auto", rasterized=True)

    cbar = fig.colorbar(img, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02)
    cbar.set_label("Water Depth (m)", fontsize=11)

    ax.set_title("GEBCO 2025 — Ocean Bathymetry\n"
                 f"({float(lats.min()):.0f}°–{float(lats.max()):.0f}°N, "
                 f"{float(lons.min()):.0f}°–{float(lons.max()):.0f}°W)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")
    print("    Key trend: deeper water (blue) towards open ocean; "
          "shelf regions clearly visible as lighter bands near coastlines.")


def _cell_area_km2(lat_deg: float, d_lat_deg: float, d_lon_deg: float) -> float:
    """Approximate area of one grid cell at a given latitude (km²)."""
    lat_rad   = np.radians(lat_deg)
    d_lat_m   = d_lat_deg * 111_320.0
    d_lon_m   = d_lon_deg * 111_320.0 * np.cos(lat_rad)
    return (d_lat_m * d_lon_m) / 1e6


def plot_foundation_histogram(ds: xr.Dataset, out_path: Path) -> None:
    """Bar chart of total ocean area (km²) per foundation type bin."""
    print("\n[5b] Plotting foundation-type area histogram ...")

    ftype  = ds["foundation_type"].values
    lats   = ds["lat"].values
    d_lat  = abs(float(lats[1] - lats[0]))
    d_lon  = abs(float(ds["lon"].values[1] - ds["lon"].values[0]))

    area_km2 = np.array(
        [_cell_area_km2(lat, d_lat, d_lon) for lat in lats]
    )[:, np.newaxis] * np.ones((1, ftype.shape[1]))

    bin_areas = []
    for i in range(len(FOUNDATION_LABELS)):
        mask = ftype == i
        bin_areas.append(float(area_km2[mask].sum()))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(FOUNDATION_LABELS, bin_areas,
                  color=FOUNDATION_COLORS, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, bin_areas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val/1e6:.2f}M km²",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Ocean Area by Foundation-Type Depth Bin",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Area (km²)", fontsize=11)
    ax.set_xlabel("Foundation Type", fontsize=11)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")

    dominant = FOUNDATION_LABELS[int(np.argmax(bin_areas))]
    print(f"    Key trend: '{dominant}' dominates by area — "
          "deep ocean constitutes the largest fraction of the survey region.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Suitability Score Placeholder
# ─────────────────────────────────────────────────────────────────────────────

def build_suitability_placeholder(ds: xr.Dataset) -> xr.Dataset:
    """
    Create an empty Suitability_Score array (NaN everywhere).
    Pre-masked so:
      - Land cells      → NaN
      - Depth > 300 m   → NaN  (Infeasible)
    Wind data will populate valid ocean cells in a later script.
    """
    print("\n[6] Building Suitability_Score placeholder ...")

    ocean   = ~ds["land_mask"].values
    shallow = ds["ocean_depth_m"].values <= 300.0

    valid_mask = ocean & shallow
    score = np.where(valid_mask, np.nan, np.nan).astype(np.float32)
    # Mark valid cells with 0.0 so the placeholder is distinguishable from
    # truly masked cells once wind data arrives
    score[valid_mask] = 0.0

    score_da = xr.DataArray(
        score, coords=ds["ocean_depth_m"].coords, name="Suitability_Score",
        attrs={
            "long_name": "Composite wind farm suitability score (0–1)",
            "units": "dimensionless",
            "note": "Populated by 03_model_training.py after wind data merge",
            "valid_min": 0.0, "valid_max": 1.0,
        },
    )
    ds = ds.assign(Suitability_Score=score_da)
    valid_cells = int(valid_mask.sum())
    print(f"    Valid placeholders (ocean ≤300 m): {valid_cells:,} cells")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Save Intermediary Parquet
# ─────────────────────────────────────────────────────────────────────────────

def save_parquet(ds: xr.Dataset, out_path: Path) -> None:
    """
    Flatten the key xarray variables to a Pandas DataFrame and save as Parquet
    for use in 02_preprocessing.py / 03_model_training.py.
    """
    print(f"\n[7] Saving intermediary parquet → {out_path} ...")

    lat2d, lon2d = np.meshgrid(ds["lat"].values, ds["lon"].values, indexing="ij")

    df = pd.DataFrame({
        "lat":              lat2d.ravel(),
        "lon":              lon2d.ravel(),
        "elevation_m":      ds["elevation"].values.ravel(),
        "ocean_depth_m":    ds["ocean_depth_m"].values.ravel(),
        "land_mask":        ds["land_mask"].values.ravel(),
        "foundation_type":  ds["foundation_type"].values.ravel(),
        "slope_deg":        ds["slope_deg"].values.ravel(),
        "slope_unstable":   ds["slope_unstable"].values.ravel(),
        "Suitability_Score":ds["Suitability_Score"].values.ravel(),
    })

    # Drop pure land rows to keep file size manageable
    df = df[~df["land_mask"]].reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"    Rows (ocean only): {len(df):,}")
    print(f"    File size        : {out_path.stat().st_size / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        # 1. Load
        ds_raw = load_bathymetry(NC_FILE)

        # 2. Preprocess
        ds = preprocess(ds_raw, stride=STRIDE)

        # 3. Foundation classification
        ds = classify_foundation(ds)

        # 4. Slope analysis
        ds = compute_slope(ds)

        # 4b. Filter to US coastal zones only
        ds = filter_us_coastal(ds)

        # 5. EDA plots
        plot_depth_heatmap(ds, OUT_DIR / "depth_heatmap.png")
        plot_foundation_histogram(ds, OUT_DIR / "foundation_histogram.png")

        # 6. Suitability placeholder
        ds = build_suitability_placeholder(ds)

        # 7. Save intermediary
        save_parquet(ds, DATA_DIR / "bathymetry_processed.parquet")

        print("\n[Done] All outputs written successfully.")
        print(f"  Plots  : {OUT_DIR}/")
        print(f"  Parquet: {DATA_DIR}/bathymetry_processed.parquet")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
