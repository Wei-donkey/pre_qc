# -*- coding: utf-8 -*-
"""
Generate neighboring stations using real geographic distance (Haversine).
Neighbors are selected within multiple radii (50km, 40km, 30km, 5km).
Results are merged into a single DataFrame with columns for each radius.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SRC_DIR = Path(__file__).resolve().parent
INPUT = SRC_DIR.parent / "data" / "gd_stations_locations.csv"
OUTPUT = SRC_DIR.parent / "data" / "gd_stations_neighbors.csv"

# Parameters
EARTH_RADIUS_KM = 6371.0
SEARCH_RADII_KM = [50.0, 40.0, 30.0, 5.0]
MAX_NEIGHBORS = 600


def build_balltree(df: pd.DataFrame):
    """
    Build BallTree using haversine distance.
    Input must be in radians: [lat, lon]
    """
    coords = np.radians(df[["lat", "lon"]].values)
    tree = BallTree(coords, metric="haversine")
    return tree, coords


def find_neighbors_multi_radius(df: pd.DataFrame):
    """
    Find neighbors at multiple search radii for each station.
    """
    tree, coords = build_balltree(df)

    # Initialize result dictionary
    results = {
        "stacode": df["stacode"].values,
    }

    # Process each radius
    for radius_km in SEARCH_RADII_KM:
        print(f"\nProcessing radius: {radius_km}km")
        search_radius_rad = radius_km / EARTH_RADIUS_KM

        neighbors_list = []
        counts_list = []

        for i, (stacode, point) in enumerate(zip(df["stacode"], coords)):
            # Query neighbors within radius
            ind, dist = tree.query_radius(
                point.reshape(1, -1),
                r=search_radius_rad,
                return_distance=True,
                sort_results=True,
            )

            indices = ind[0]
            distances = dist[0] * EARTH_RADIUS_KM  # convert to km

            # Remove itself (distance = 0)
            mask = indices != i
            indices = indices[mask]
            distances = distances[mask]

            # If too many neighbors → keep closest MAX_NEIGHBORS
            if len(indices) > MAX_NEIGHBORS:
                indices = indices[:MAX_NEIGHBORS]
                distances = distances[:MAX_NEIGHBORS]

            neighbors_ids = df.iloc[indices]["stacode"].astype(str).tolist()
            neighbors_str = ",".join(neighbors_ids)

            neighbors_list.append(neighbors_str)
            counts_list.append(len(neighbors_ids))

            if i % 100 == 0 or i == len(df) - 1:
                print(f"  {i+1}/{df.shape[0]}: Processed {stacode}: {len(neighbors_ids)} neighbors")

        # Store results for this radius
        suffix = str(int(radius_km))
        results[f"neighbors{suffix}"] = neighbors_list
        results[f"count{suffix}"] = counts_list

    return pd.DataFrame(results)


def main() -> None:
    print(f"Reading station locations from: {INPUT}")
    df = pd.read_csv(INPUT, encoding="utf-8-sig")

    neighbors_df = find_neighbors_multi_radius(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    neighbors_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\nWrote {len(neighbors_df)} rows to {OUTPUT}")
    print("\nColumn structure:")
    print(neighbors_df.columns.tolist())


if __name__ == "__main__":
    main()
