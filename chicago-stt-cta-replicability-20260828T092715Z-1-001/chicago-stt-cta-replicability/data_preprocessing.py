"""
data_preprocessing.py — Data loading, cleaning, grid construction, and fusion.

Reads the three input CSVs, builds a spatial grid over The Loop,
maps traffic segments and TNP trips to grid cells, fuses into a
4D tensor (T, H, W, F), and saves to disk.
"""

import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
import holidays

import config

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Parse Loop Boundary
# ═══════════════════════════════════════════════════════════════════════════════

def load_loop_polygon():
    """Load the Loop community area polygon from the boundaries CSV."""
    print("[1/6] Loading Loop boundary polygon …")
    df = pd.read_csv(config.BOUNDARIES_CSV, decimal=",")
    # Defensive: filter to Community Area 32 (Loop) in case file has multiple rows
    loop_rows = df[df["AREA_NUMBE"].astype(str).str.strip() == "32"]
    if len(loop_rows) == 0:
        loop_rows = df  # fallback: single-row file
    geom_str = loop_rows.iloc[0]["the_geom"]
    polygon = wkt.loads(geom_str)
    # Extract exact bounding box
    minlon, minlat, maxlon, maxlat = polygon.bounds
    bbox = {
        "min_lat": minlat,
        "max_lat": maxlat,
        "min_lon": minlon,
        "max_lon": maxlon,
    }
    print(f"    Bounding box: lat [{minlat:.4f}, {maxlat:.4f}], "
          f"lon [{minlon:.4f}, {maxlon:.4f}]")
    return polygon, bbox


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Build Spatial Grid
# ═══════════════════════════════════════════════════════════════════════════════

def build_grid(bbox, cell_size_m=None):
    """Build a regular lat/lon grid over the bounding box.

    Returns
    -------
    n_rows, n_cols : int
        Grid dimensions.
    lat_edges, lon_edges : np.ndarray
        Cell boundary coordinates (length n_rows+1, n_cols+1).
    """
    if cell_size_m is None:
        cell_size_m = config.GRID_CELL_SIZE_M

    n_rows, n_cols = config.compute_grid_dims(bbox, cell_size_m)

    lat_edges = np.linspace(bbox["min_lat"], bbox["max_lat"], n_rows + 1)
    lon_edges = np.linspace(bbox["min_lon"], bbox["max_lon"], n_cols + 1)

    print(f"[2/6] Grid: {n_rows} rows × {n_cols} cols = {n_rows * n_cols} cells "
          f"({cell_size_m}m cells)")
    return n_rows, n_cols, lat_edges, lon_edges


def assign_to_grid(lat, lon, lat_edges, lon_edges):
    """Assign a (lat, lon) point to a grid cell.

    Returns (row, col) indices, or (None, None) if outside grid.
    """
    if lat < lat_edges[0] or lat > lat_edges[-1]:
        return None, None
    if lon < lon_edges[0] or lon > lon_edges[-1]:
        return None, None
    row = min(int(np.searchsorted(lat_edges, lat, side="right")) - 1,
              len(lat_edges) - 2)
    col = min(int(np.searchsorted(lon_edges, lon, side="right")) - 1,
              len(lon_edges) - 2)
    return row, col


def assign_to_grid_vectorized(lats, lons, lat_edges, lon_edges):
    """Vectorized grid assignment for arrays of coordinates.

    Returns arrays of (row, col). Values are -1 where outside grid.
    """
    rows = np.searchsorted(lat_edges, lats, side="right").astype(int) - 1
    cols = np.searchsorted(lon_edges, lons, side="right").astype(int) - 1

    n_rows = len(lat_edges) - 1
    n_cols = len(lon_edges) - 1

    # Clamp and mark invalid
    valid = (
        (lats >= lat_edges[0]) & (lats <= lat_edges[-1]) &
        (lons >= lon_edges[0]) & (lons <= lon_edges[-1])
    )
    rows = np.clip(rows, 0, n_rows - 1)
    cols = np.clip(cols, 0, n_cols - 1)
    rows[~valid] = -1
    cols[~valid] = -1
    return rows, cols


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Process Traffic Tracker Data
# ═══════════════════════════════════════════════════════════════════════════════

def process_traffic_data(lat_edges, lon_edges):
    """Load, clean, and aggregate Chicago Traffic Tracker data.

    Returns a DataFrame with columns:
    [hour, grid_row, grid_col, avg_speed, min_speed, num_segments, bus_count]
    """
    print("[3/6] Processing Traffic Tracker data …")

    agg_frames = []
    total_rows = 0
    kept_rows = 0

    for i, chunk in enumerate(pd.read_csv(
        config.TRAFFIC_CSV,
        decimal=",",
        thousands=" ",
        chunksize=config.CHUNK_SIZE,
        dtype={
            "SEGMENT_ID": "int32",
            "SPEED": "float32",
            "BUS_COUNT": "int32",
        },
        usecols=["TIME", "SEGMENT_ID", "SPEED", "BUS_COUNT",
                 "START_LATITUDE", "START_LONGITUDE",
                 "END_LATITUDE", "END_LONGITUDE"],
    )):
        total_rows += len(chunk)

        # Filter out SPEED == -1 (no data)
        chunk = chunk[chunk["SPEED"] > 0].copy()

        # Parse timestamps
        chunk["TIME"] = pd.to_datetime(chunk["TIME"], format="%m/%d/%Y %I:%M:%S %p")

        # Compute segment midpoint
        chunk["mid_lat"] = (chunk["START_LATITUDE"] + chunk["END_LATITUDE"]) / 2.0
        chunk["mid_lon"] = (chunk["START_LONGITUDE"] + chunk["END_LONGITUDE"]) / 2.0

        # Assign to grid
        rows, cols = assign_to_grid_vectorized(
            chunk["mid_lat"].values, chunk["mid_lon"].values,
            lat_edges, lon_edges
        )
        chunk["grid_row"] = rows
        chunk["grid_col"] = cols

        # Keep only points inside the grid
        chunk = chunk[(chunk["grid_row"] >= 0) & (chunk["grid_col"] >= 0)]
        kept_rows += len(chunk)

        # Floor to hour
        chunk["hour"] = chunk["TIME"].dt.floor("h")

        # Aggregate per (hour, grid_row, grid_col)
        agg = chunk.groupby(["hour", "grid_row", "grid_col"]).agg(
            avg_speed=("SPEED", "mean"),
            min_speed=("SPEED", "min"),
            num_segments=("SEGMENT_ID", "nunique"),
            bus_count=("BUS_COUNT", "sum"),
        ).reset_index()
        agg_frames.append(agg)

        if (i + 1) % 5 == 0:
            print(f"    … processed {total_rows:,} traffic rows")

    traffic_agg = pd.concat(agg_frames, ignore_index=True)

    # Re-aggregate across chunks using weighted mean for speed
    # (mean-of-means is incorrect when chunks have different counts)
    traffic_agg["speed_sum"] = traffic_agg["avg_speed"] * traffic_agg["num_segments"]
    traffic_agg = traffic_agg.groupby(["hour", "grid_row", "grid_col"]).agg(
        speed_sum=("speed_sum", "sum"),
        min_speed=("min_speed", "min"),
        num_segments=("num_segments", "sum"),
        bus_count=("bus_count", "sum"),
    ).reset_index()
    traffic_agg["avg_speed"] = traffic_agg["speed_sum"] / traffic_agg["num_segments"].clip(lower=1)
    traffic_agg.drop(columns=["speed_sum"], inplace=True)

    print(f"    Total rows: {total_rows:,} → kept in Loop grid: {kept_rows:,}")
    print(f"    Aggregated records: {len(traffic_agg):,}")
    return traffic_agg


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Process TNP (Ride-Hailing) Data
# ═══════════════════════════════════════════════════════════════════════════════

def process_tnp_data(lat_edges, lon_edges):
    """Load, clean, and aggregate TNP trip data.

    Returns a DataFrame with columns:
    [hour, grid_row, grid_col, pickup_count, dropoff_count,
     avg_trip_duration, avg_trip_distance]
    """
    print("[4/6] Processing TNP (ride-hailing) data …")

    pickup_frames = []
    dropoff_frames = []
    total_rows = 0
    kept_pickups = 0
    kept_dropoffs = 0

    for i, chunk in enumerate(pd.read_csv(
        config.TNP_CSV,
        decimal=",",
        chunksize=config.CHUNK_SIZE,
        usecols=[
            "Trip Start Timestamp",
            "Trip Seconds", "Trip Miles",
            "Pickup Centroid Latitude", "Pickup Centroid Longitude",
            "Dropoff Centroid Latitude", "Dropoff Centroid Longitude",
        ],
        dtype={
            "Trip Seconds": "str",
            "Trip Miles": "str",
        },
    )):
        total_rows += len(chunk)

        # Ensure numeric types for Trip Seconds / Trip Miles
        # Trip Seconds: integer-like, may have \u202f (narrow no-break space) as thousands separator
        # Trip Miles: may use comma as decimal separator
        for col in ["Trip Seconds", "Trip Miles"]:
            chunk[col] = (
                chunk[col]
                .astype(str)
                .str.replace("\u202f", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)   # comma-decimal → period
            )
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Parse timestamps
        chunk["Trip Start Timestamp"] = pd.to_datetime(
            chunk["Trip Start Timestamp"], format="%m/%d/%Y %I:%M:%S %p"
        )
        chunk["hour"] = chunk["Trip Start Timestamp"].dt.floor("h")

        # ── Pickups ──
        pu = chunk.dropna(subset=["Pickup Centroid Latitude", "Pickup Centroid Longitude"]).copy()
        pu_rows, pu_cols = assign_to_grid_vectorized(
            pu["Pickup Centroid Latitude"].values,
            pu["Pickup Centroid Longitude"].values,
            lat_edges, lon_edges,
        )
        pu["grid_row"] = pu_rows
        pu["grid_col"] = pu_cols
        pu = pu[(pu["grid_row"] >= 0) & (pu["grid_col"] >= 0)]
        kept_pickups += len(pu)

        pu_agg = pu.groupby(["hour", "grid_row", "grid_col"]).agg(
            pickup_count=("Trip Start Timestamp", "size"),
            avg_trip_duration=("Trip Seconds", "mean"),
            avg_trip_distance=("Trip Miles", "mean"),
        ).reset_index()
        pickup_frames.append(pu_agg)

        # ── Dropoffs ──
        do = chunk.dropna(subset=["Dropoff Centroid Latitude", "Dropoff Centroid Longitude"]).copy()
        do_rows, do_cols = assign_to_grid_vectorized(
            do["Dropoff Centroid Latitude"].values,
            do["Dropoff Centroid Longitude"].values,
            lat_edges, lon_edges,
        )
        do["grid_row"] = do_rows
        do["grid_col"] = do_cols
        do = do[(do["grid_row"] >= 0) & (do["grid_col"] >= 0)]
        kept_dropoffs += len(do)

        do_agg = do.groupby(["hour", "grid_row", "grid_col"]).agg(
            dropoff_count=("Trip Start Timestamp", "size"),
        ).reset_index()
        dropoff_frames.append(do_agg)

        if (i + 1) % 5 == 0:
            print(f"    … processed {total_rows:,} TNP rows")

    # Combine pickup aggregations with weighted means for duration/distance
    pickups = pd.concat(pickup_frames, ignore_index=True)
    pickups["dur_sum"] = pickups["avg_trip_duration"] * pickups["pickup_count"]
    pickups["dist_sum"] = pickups["avg_trip_distance"] * pickups["pickup_count"]
    pickups = pickups.groupby(["hour", "grid_row", "grid_col"]).agg(
        pickup_count=("pickup_count", "sum"),
        dur_sum=("dur_sum", "sum"),
        dist_sum=("dist_sum", "sum"),
    ).reset_index()
    pickups["avg_trip_duration"] = pickups["dur_sum"] / pickups["pickup_count"].clip(lower=1)
    pickups["avg_trip_distance"] = pickups["dist_sum"] / pickups["pickup_count"].clip(lower=1)
    pickups.drop(columns=["dur_sum", "dist_sum"], inplace=True)

    # Combine dropoff aggregations
    dropoffs = pd.concat(dropoff_frames, ignore_index=True)
    dropoffs = dropoffs.groupby(["hour", "grid_row", "grid_col"]).agg(
        dropoff_count=("dropoff_count", "sum"),
    ).reset_index()

    # Merge pickups and dropoffs
    tnp_agg = pd.merge(pickups, dropoffs,
                       on=["hour", "grid_row", "grid_col"],
                       how="outer")

    print(f"    Total rows: {total_rows:,}")
    print(f"    Pickups in Loop: {kept_pickups:,}, Dropoffs in Loop: {kept_dropoffs:,}")
    print(f"    Aggregated records: {len(tnp_agg):,}")
    return tnp_agg


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Fuse into Grid Tensor
# ═══════════════════════════════════════════════════════════════════════════════

def build_fused_tensor(traffic_agg, tnp_agg, n_rows, n_cols):
    """Fuse traffic and TNP data into a 4D numpy array.

    Returns
    -------
    tensor : np.ndarray, shape (T, H, W, F)
    time_index : pd.DatetimeIndex
    """
    print("[5/6] Building fused grid tensor …")

    # Create complete hourly time index for all of 2025
    time_index = pd.date_range("2025-01-01", "2025-12-31 23:00:00", freq="h")
    T = len(time_index)
    F = config.NUM_FEATURES
    print(f"    Time steps: {T}, Grid: {n_rows}×{n_cols}, Features: {F}")

    # Initialize tensor
    tensor = np.zeros((T, n_rows, n_cols, F), dtype=np.float32)

    # Create hour → index mapping
    hour_to_idx = {h: i for i, h in enumerate(time_index)}

    # ── Fill traffic features (vectorized) ──
    print("    Filling traffic features …")
    traffic_agg = traffic_agg.copy()
    traffic_agg["t_idx"] = traffic_agg["hour"].map(hour_to_idx)
    traffic_agg = traffic_agg.dropna(subset=["t_idx"])
    traffic_agg["t_idx"] = traffic_agg["t_idx"].astype(int)
    mask = (
        (traffic_agg["grid_row"] >= 0) & (traffic_agg["grid_row"] < n_rows) &
        (traffic_agg["grid_col"] >= 0) & (traffic_agg["grid_col"] < n_cols)
    )
    traffic_agg = traffic_agg[mask]
    t_idx = traffic_agg["t_idx"].values
    r_idx = traffic_agg["grid_row"].astype(int).values
    c_idx = traffic_agg["grid_col"].astype(int).values
    for feat_name in ["avg_speed", "min_speed", "num_segments", "bus_count"]:
        f_idx = config.ALL_FEATURES.index(feat_name)
        tensor[t_idx, r_idx, c_idx, f_idx] = traffic_agg[feat_name].values

    # ── Fill TNP features (vectorized) ──
    print("    Filling TNP features …")
    tnp_agg = tnp_agg.copy()
    tnp_agg["t_idx"] = tnp_agg["hour"].map(hour_to_idx)
    tnp_agg = tnp_agg.dropna(subset=["t_idx"])
    tnp_agg["t_idx"] = tnp_agg["t_idx"].astype(int)
    mask = (
        (tnp_agg["grid_row"] >= 0) & (tnp_agg["grid_row"] < n_rows) &
        (tnp_agg["grid_col"] >= 0) & (tnp_agg["grid_col"] < n_cols)
    )
    tnp_agg = tnp_agg[mask]
    tnp_agg = tnp_agg.fillna(0)
    t_idx = tnp_agg["t_idx"].values
    r_idx = tnp_agg["grid_row"].astype(int).values
    c_idx = tnp_agg["grid_col"].astype(int).values
    for feat_name in ["pickup_count", "dropoff_count", "avg_trip_duration", "avg_trip_distance"]:
        f_idx = config.ALL_FEATURES.index(feat_name)
        if feat_name in tnp_agg.columns:
            tensor[t_idx, r_idx, c_idx, f_idx] = tnp_agg[feat_name].values

    # ── Fill temporal features ──
    print("    Filling temporal features …")
    # Build US holiday calendar for Illinois (covers federal + state holidays)
    us_holidays = holidays.US(years=2025, state='IL')
    for i, ts in enumerate(time_index):
        tensor[i, :, :, config.ALL_FEATURES.index("hour_of_day")] = ts.hour
        tensor[i, :, :, config.ALL_FEATURES.index("day_of_week")] = ts.dayofweek
        tensor[i, :, :, config.ALL_FEATURES.index("is_weekend")] = int(ts.dayofweek >= 5)
        tensor[i, :, :, config.ALL_FEATURES.index("month")] = ts.month
        tensor[i, :, :, config.ALL_FEATURES.index("is_holiday")] = int(ts.date() in us_holidays)

    # ── Fill weather features ──
    print("    Filling weather features …")
    weather_path = config.WEATHER_CSV
    if os.path.exists(weather_path):
        weather_df = pd.read_csv(weather_path, parse_dates=["datetime"])
        weather_df = weather_df.set_index("datetime").sort_index()
        # Drop duplicate timestamps (keep first — can happen with DST)
        weather_df = weather_df[~weather_df.index.duplicated(keep="first")]
        # Map each hour to its weather values (uniform across all cells)
        temp_idx = config.ALL_FEATURES.index("temperature_2m")
        precip_idx = config.ALL_FEATURES.index("precipitation")
        matched = 0
        for i, ts in enumerate(time_index):
            if ts in weather_df.index:
                tensor[i, :, :, temp_idx] = weather_df.loc[ts, "temperature_2m"]
                tensor[i, :, :, precip_idx] = weather_df.loc[ts, "precipitation"]
                matched += 1
        print(f"    Matched {matched}/{T} hours with weather data")
    else:
        print(f"    WARNING: Weather file not found at {weather_path}")
        print(f"    Run fetch_weather.py first. Weather features will be zeros.")

    # ── Handle missing speed data ──
    # For cells with 0 speed (no data), interpolate along time axis
    speed_idx = config.ALL_FEATURES.index("avg_speed")
    min_speed_idx = config.ALL_FEATURES.index("min_speed")
    for r in range(n_rows):
        for c in range(n_cols):
            for feat_idx in [speed_idx, min_speed_idx]:
                series = tensor[:, r, c, feat_idx]
                if np.any(series > 0):
                    # Replace 0s with NaN, then interpolate
                    mask = series == 0
                    if mask.all():
                        continue
                    series[mask] = np.nan
                    # Forward-fill then backward-fill
                    s = pd.Series(series)
                    s = s.ffill().bfill()
                    tensor[:, r, c, feat_idx] = s.values

    print(f"    Tensor shape: {tensor.shape}")
    print(f"    Non-zero speed cells: {(tensor[:, :, :, speed_idx] > 0).sum():,}")
    print(f"    Total pickups: {tensor[:, :, :, config.ALL_FEATURES.index('pickup_count')].sum():,.0f}")
    print(f"    Total dropoffs: {tensor[:, :, :, config.ALL_FEATURES.index('dropoff_count')].sum():,.0f}")

    return tensor, time_index


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Compute Normalization Statistics & Save
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_save(tensor, time_index, n_rows, n_cols, bbox):
    """Compute train-set normalization stats and save everything to disk."""
    print("[6/6] Computing normalization statistics and saving …")

    # Determine training indices
    train_end = pd.Timestamp(config.TRAIN_END)
    train_mask = time_index <= train_end
    train_indices = np.where(train_mask)[0]
    train_data = tensor[train_indices]

    # Compute per-feature mean and std from training data
    # Shape: (F,)
    mean = train_data.reshape(-1, config.NUM_FEATURES).mean(axis=0)
    std = train_data.reshape(-1, config.NUM_FEATURES).std(axis=0)
    std[std < 1e-8] = 1.0  # Avoid division by zero

    scaler = {"mean": mean, "std": std}

    # Save tensor
    tensor_path = os.path.join(config.TENSOR_DIR, "grid_tensor.npy")
    np.save(tensor_path, tensor)
    print(f"    Saved tensor to {tensor_path}")

    # Save time index
    time_path = os.path.join(config.TENSOR_DIR, "time_index.pkl")
    with open(time_path, "wb") as f:
        pickle.dump(time_index, f)
    print(f"    Saved time index to {time_path}")

    # Save scaler
    scaler_path = os.path.join(config.TENSOR_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Saved scaler to {scaler_path}")

    # Save grid metadata
    meta = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "bbox": bbox,
        "cell_size_m": config.GRID_CELL_SIZE_M,
        "features": config.ALL_FEATURES,
    }
    meta_path = os.path.join(config.TENSOR_DIR, "grid_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"    Saved grid metadata to {meta_path}")

    # Print summary stats
    print("\n    === Feature Statistics (Training Set) ===")
    for i, feat in enumerate(config.ALL_FEATURES):
        print(f"    {feat:20s}  mean={mean[i]:10.3f}  std={std[i]:10.3f}")

    return scaler


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    config.print_config()

    # 1. Load boundary
    polygon, bbox = load_loop_polygon()

    # Update config bbox with exact values
    config.LOOP_BBOX = bbox
    n_rows, n_cols = config.compute_grid_dims(bbox)
    # Update config
    config.GRID_ROWS = n_rows
    config.GRID_COLS = n_cols

    # 2. Build grid
    n_rows, n_cols, lat_edges, lon_edges = build_grid(bbox)

    # 3. Process traffic data
    traffic_agg = process_traffic_data(lat_edges, lon_edges)

    # 4. Process TNP data
    tnp_agg = process_tnp_data(lat_edges, lon_edges)

    # 5. Build fused tensor
    tensor, time_index = build_fused_tensor(traffic_agg, tnp_agg, n_rows, n_cols)

    # 6. Save
    scaler = compute_and_save(tensor, time_index, n_rows, n_cols, bbox)

    print("\n✓ Data preprocessing complete!")
    return tensor, time_index, scaler


if __name__ == "__main__":
    main()
