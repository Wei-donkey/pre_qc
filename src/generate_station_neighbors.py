# -*- coding: utf-8 -*-
"""
Generate neighboring stations using real geographic distance (Haversine).
Neighbors are selected within a given radius (km). If count exceeds limit,
only the closest MAX_NEIGHBORS are kept.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SRC_DIR = Path(__file__).resolve().parent
INPUT = SRC_DIR.parent / 'data' / 'gd_stations_locations.csv'
OUTPUT = SRC_DIR.parent / 'data' / 'gd_stations_neighbors.csv'

# Parameters
EARTH_RADIUS_KM = 6371.0
SEARCH_RADIUS_KM = 50.0
MAX_NEIGHBORS = 600


def build_balltree(df: pd.DataFrame):
    """
    Build BallTree using haversine distance.
    Input must be in radians: [lat, lon]
    """
    coords = np.radians(df[['lat', 'lon']].values)
    tree = BallTree(coords, metric='haversine')
    return tree, coords


def find_neighbors(df: pd.DataFrame):
    tree, coords = build_balltree(df)

    search_radius_rad = SEARCH_RADIUS_KM / EARTH_RADIUS_KM

    results = []

    for i, (stacode, point) in enumerate(zip(df['stacode'], coords)):

        # Query neighbors within radius
        ind, dist = tree.query_radius(
            point.reshape(1, -1),
            r=search_radius_rad,
            return_distance=True,
            sort_results=True
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

        neighbors_ids = df.iloc[indices]['stacode'].astype(str).tolist()
        neighbors_str = ",".join(neighbors_ids)

        results.append({
            'stacode': stacode,
            'neighbors': neighbors_str,
            'count': len(neighbors_ids)
        })

        print(f"{i+1}/{df.shape[0]}: Processed {stacode}: {len(neighbors_ids)} neighbors")

    return pd.DataFrame(results)


def main() -> None:
    print(f"Reading station locations from: {INPUT}")
    df = pd.read_csv(INPUT, encoding='utf-8-sig')

    neighbors_df = find_neighbors(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    neighbors_df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

    print(f"Wrote {len(neighbors_df)} rows to {OUTPUT}")


if __name__ == '__main__':
    main()