# -*- coding: utf-8 -*-
"""
Generate neighboring stations using real geographic distance (Haversine).
Neighbors are selected within a given radius (km). 
Remove MAX_NEIGHBORS = 600 to ensure all stations within given radius are included.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SRC_DIR = Path(__file__).resolve().parent
INPUT_STATIONS = SRC_DIR.parent / 'data' / 'gd_stations_locations.csv'
OUTPUT_STATION_NEIGHBORS = SRC_DIR.parent / 'data' / 'gd_stations_neighbors.csv'

# Parameters
EARTH_RADIUS_KM = 6371.0
SEARCH_RADIUS_KM = 50.0


def build_balltree(df_stations: pd.DataFrame):
    """
    Build BallTree using haversine distance.
    Input must be in radians: [lat, lon]
    """
    # Convert [lat, lon] to radians for Haversine metric
    coords = np.radians(df_stations[['lat', 'lon']].values)
    tree = BallTree(coords, metric='haversine')
    return tree, coords


def find_neighbors(df_stations: pd.DataFrame, tree, coords: np.ndarray):
    """  Iterate through station points-coords and find neighboring stations within SEARCH_RADIUS_KM.  """

    search_radius_rad = SEARCH_RADIUS_KM / EARTH_RADIUS_KM
    total_stations = df_stations.shape[0]

    results = []

    for i, (stacode, rad_point) in enumerate(zip(df_stations['stacode'], coords)):

        # Query neighbors within radius
        ind, dist = tree.query_radius(
            rad_point.reshape(1, -1),
            r=search_radius_rad,
            return_distance=True,
            sort_results=True
        )

        indices = ind[0]
        if len(indices) == 0:
            continue
        # distances = dist[0] * EARTH_RADIUS_KM  # convert to km

        # Remove itself (distance = 0)
        mask = indices != i
        indices = indices[mask]

        lst_neighbors = df_stations.iloc[indices]['stacode'].astype(str).tolist()
        neighbors_str = ",".join(lst_neighbors)

        results.append({
            'stacode': stacode,
            f"neighbors{int(SEARCH_RADIUS_KM)}": neighbors_str,
            'count': len(lst_neighbors)
        })

        print(f"{i+1}/{total_stations}: Processed {stacode}: {len(lst_neighbors)} neighbors")

    return pd.DataFrame(results)


def main() -> None:
    print(f"Reading station locations from: {INPUT_STATIONS}")
    df_stations = pd.read_csv(INPUT_STATIONS, encoding='utf-8-sig')

    tree, coords = build_balltree(df_stations)
    df_neighbors = find_neighbors(df_stations, tree, coords)

    OUTPUT_STATION_NEIGHBORS.parent.mkdir(parents=True, exist_ok=True)
    df_neighbors.to_csv(OUTPUT_STATION_NEIGHBORS, index=False, encoding='utf-8-sig')

    print(f"Wrote {len(df_neighbors)} rows to {OUTPUT_STATION_NEIGHBORS}")


if __name__ == '__main__':
    main()