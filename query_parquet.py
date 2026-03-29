"""
query_parquet.py — Ad-hoc queries on bathymetry_processed.parquet
=================================================================
Run:  python query_parquet.py
"""

import pandas as pd

PARQUET = "data/bathymetry_processed.parquet"
FOUNDATION_LABELS = {
    0: "Monopile (0–15 m)",
    1: "Jacket/Fixed (15–60 m)",
    2: "Floating (60–300 m)",
    3: "Infeasible (>300 m)",
}

df = pd.read_parquet(PARQUET)

# ── Quick schema check ────────────────────────────────────────────────────────
print("=== Schema ===")
print(df.dtypes)
print(f"\nTotal rows (ocean only): {len(df):,}")

# ── Suitability_Score null breakdown ─────────────────────────────────────────
print("\n=== Suitability_Score null breakdown ===")
null_count    = df["Suitability_Score"].isna().sum()
nonnull_count = df["Suitability_Score"].notna().sum()
print(f"  Non-null (valid placeholder) : {nonnull_count:,}")
print(f"  Null  (Infeasible / >300 m)  : {null_count:,}")
print()
print("NOTE: Null cells are deep-water (foundation_type == 3, depth > 300 m).")
print("      Non-null cells are set to 0.0 as placeholder, ready for wind data.")

# ── Head of feasible ocean rows (depth 0–300 m) ─────────────────────────────
print("\n=== Head: feasible ocean rows (depth 0–300 m) ===")
feasible = df[(df["ocean_depth_m"] >= 0) & (df["ocean_depth_m"] <= 300)].copy()
feasible["foundation_label"] = feasible["foundation_type"].map(FOUNDATION_LABELS)
print(f"  Total feasible rows: {len(feasible):,}")
print()
print(feasible[[
    "lat", "lon", "ocean_depth_m", "foundation_label",
    "slope_deg", "slope_unstable"
]].head(10).to_string(index=False))

# ── Cross-tab: foundation type vs. null/non-null ──────────────────────────────
print("\n=== foundation_type distribution (all ocean rows) ===")
grp = df.groupby("foundation_type", dropna=False)["Suitability_Score"]
xtab = pd.DataFrame({
    "total_rows": grp.size(),
    "non_null":   grp.apply(lambda s: s.notna().sum()),
}).assign(null=lambda x: x["total_rows"] - x["non_null"])
xtab.index = xtab.index.map(lambda i: FOUNDATION_LABELS.get(int(i), str(i)) if pd.notna(i) else "Unknown")
print(xtab.to_string())
