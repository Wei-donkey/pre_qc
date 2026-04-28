# -*- coding: utf-8 -*-
"""
Plot precipitation events over 100mm against the data from neighboring stations as spatial scatter maps.
The script is organized in three major blocks:
1. data loading,
2. database access,
3. plotting.
@author: Wei Liu
"""

from __future__ import annotations

import configparser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


SRC_DIR = Path(__file__).resolve().parent
PROJ_DIR = SRC_DIR.parent
DATA_DIR = PROJ_DIR / 'data'
OUTPUT_DIR_FALSE = PROJ_DIR / 'figures' / 'false_precip_maps'
OUTPUT_DIR_TRUE = PROJ_DIR / 'figures' / 'true_precip_maps'
OUTPUT_DIR_UNCERTAIN = PROJ_DIR / 'figures' / 'uncertain_precip_maps'
OUTPUT_DIR_FALSE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_TRUE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_UNCERTAIN.mkdir(parents=True, exist_ok=True)

PRECIP_FILE = DATA_DIR / 'gd_precip_hourly_2003-2025.csv'
NEIGHBOR_FILE = DATA_DIR / 'gd_stations_neighbors.csv'
LOCATION_FILE = DATA_DIR / 'gd_stations_locations.csv'
CONFIG_FILE = SRC_DIR / 'config_db.ini'

STYLE_SETTINGS = {
    'style': 'seaborn-v0_8-darkgrid',
    'grid.linewidth': 0.5,
    'font.size': 6,
    'lines.linewidth': 1.0,
    'figure.titlesize': 8,
    'figure.dpi': 300,
    'legend.fontsize': 6,
    'legend.frameon': True,
    'legend.framealpha': 0.5,
    'legend.facecolor': 'inherit',
    'legend.edgecolor': 'white',
}


def set_plot_style() -> None:
    for key, value in STYLE_SETTINGS.items():
        if key == 'style':
            mpl.style.use(STYLE_SETTINGS['style'])
        else:
            mpl.rcParams[key] = value


def load_center_r() -> pd.DataFrame:
    precip = pd.read_csv(PRECIP_FILE, dtype=str)
    precip['r'] = pd.to_numeric(precip['r'], errors='coerce')
    label_map = {0:'False', 1:'True',9:'Uncertain'}
    precip['label'] = precip['datatype'].astype(int).map(label_map)
    precip['ddatetime'] = pd.to_datetime(precip['ddatetime'], errors='coerce')
    precip = precip.dropna(subset=['stacode', 'ddatetime', 'r']).copy()
    precip['year'] = precip['ddatetime'].dt.year
    precip['month'] = precip['ddatetime'].dt.month
    precip = precip.sort_values(['ddatetime', 'r'], ascending=[True, False])
    precip = precip[precip['r'] >= 100].reset_index(drop=True).copy()  # Filter for events with r >= 100mm
    # precip = precip.iloc[0:10]  # For testing with a smaller subset of events
    # precip = precip[precip['stacode']=='G2109'].reset_index(drop=True).copy() 
    return precip


def load_loc() -> pd.DataFrame:
    loc_df = pd.read_csv(LOCATION_FILE, header=0, dtype=str, quotechar='"', engine='python',
        names=['stacode', 'lon', 'lat', 'elevation', 'station_type'],
        )
    loc_df = loc_df.dropna(subset=['stacode', 'lon', 'lat']).copy()
    loc_df['stacode'] = loc_df['stacode'].astype(str).str.strip()
    loc_df['lon'] = pd.to_numeric(loc_df['lon'], errors='coerce')
    loc_df['lat'] = pd.to_numeric(loc_df['lat'], errors='coerce')
    loc_df = loc_df.dropna(subset=['lon', 'lat'])
    loc_df = loc_df[['stacode', 'lon', 'lat']].drop_duplicates('stacode')

    return loc_df


def load_neighbors() -> dict[str, list[str]]:
    neighbors_df = pd.read_csv(NEIGHBOR_FILE, header=0, dtype=str, quotechar='"', engine='python',
        names=['stacode', 'neighbors', 'count', 'updated'],
        )
    neighbors_df = neighbors_df.dropna(subset=['stacode']).copy()
    neighbors_df['stacode'] = neighbors_df['stacode'].astype(str).str.strip()
    neighbors_df['neighbors'] = neighbors_df['neighbors'].fillna('').astype(str)
    neighbors_df['neighbors_list'] = neighbors_df['neighbors'].str.split(',').apply(lambda x: [code.strip() for code in x if code.strip()])
    neighbor_map = neighbors_df.set_index('stacode')['neighbors_list'].to_dict()

    return neighbor_map


def make_db_engine():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE, encoding='utf-8-sig')
    db = config['CROSS_WEATHER']
    user = db['user']
    password = quote(db['password'])
    host = db['host']
    port = db['port']
    service = db['service']
    return create_engine(f"oracle+oracledb://{user}:{password}@{host}:{port}/{service}", echo=False)


def plot_event(
    center_lon: float,
    center_lat: float,
    center_r: float,
    event_time: pd.Timestamp,
    neighbor_gdf: gpd.GeoDataFrame,
    output_path: Path,
    center_code: str,
    event_label: str,
) -> None:
    extent = [center_lon - 0.46, center_lon + 0.46, center_lat - 0.46, center_lat + 0.46]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='#f0f0f0')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='face', facecolor='#e6f0ff')
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none')
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black', facecolor='none')
    ax.add_feature(land, zorder=0)
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(coastline, linewidth=0.5, zorder=1)
    ax.add_feature(borders, linewidth=0.5, alpha=0.5, zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    x_start = np.ceil(extent[0] * 10) / 10
    y_start = np.ceil(extent[2] * 10) / 10
    x_end = np.ceil(extent[1] * 10) / 10
    y_end = np.ceil(extent[3] * 10) / 10
    gl.xlocator = mticker.FixedLocator(np.arange(x_start, x_end + 0.1, 0.1))
    gl.ylocator = mticker.FixedLocator(np.arange(y_start, y_end + 0.1, 0.1))

    if not neighbor_gdf.empty:
        vmax = float(max(neighbor_gdf['r'].max(), 0.1))
        scatter = ax.scatter(
            neighbor_gdf.geometry.x,
            neighbor_gdf.geometry.y,
            c=neighbor_gdf['r'],
            cmap='viridis',
            vmin=0,
            vmax=vmax,
            s=25,
            # edgecolor='black',
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('precipitation (mm)', fontsize=6)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        large_values = neighbor_gdf[neighbor_gdf['r'] >= 10]
        for _, row in large_values.iterrows():
            ax.text(
                row.geometry.x + 0.01,
                # row.geometry.y + 0.01,
                f"{row['r']:.1f}",
                transform=ccrs.PlateCarree(),
                fontsize=5,
                color='black',
                va='bottom',
                ha='left',
                zorder=3,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5),
            )

    ax.scatter(
        center_lon,
        center_lat,
        marker='*',
        c='red',
        s=80,
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
        zorder=3,
        label=f"{event_label} precipitation @ {center_code}",
    )
    ax.text(
        center_lon + 0.025,
        center_lat,
        f"{center_r}",
        transform=ccrs.PlateCarree(),
        fontsize=6,
        color='red',
        va='center',
        ha='left',
        zorder=4,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1),
    )
    ax.set_title(
        f"{event_label} precipitation: {center_code} @ {event_time.strftime('%Y-%m-%d %H:%M')}\n"
        f"Neighboring station number: {len(neighbor_gdf)}",
        fontsize=8,
    )
    ax.legend(loc='lower left', framealpha=0.8)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    set_plot_style()

    print('Loading precipitation and station metadata...')
    precip, neighbor_map, loc_df = load_center_r(), load_neighbors(), load_loc()
    print(f"Loaded {len(precip)} event(s), {len(neighbor_map)} neighbor station entries, {len(loc_df)} station locations.")

    precip = precip.copy()

    print('Connecting to database...')
    engine = make_db_engine()
    print('Database connection established.')

    for idx, event in precip.iterrows():
        print(f"Processing event {idx+1}/{len(precip)}: {event['stacode']} at {event['ddatetime']}")
        year = int(event['year'])
        event_time = event['ddatetime']

        dt_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
        sql_awst = (
            f"select stacode, ddatetime, r from awst_cli_mul_hor_{year} "
            f"where length(stacode)=5 "
            f"and ddatetime = TO_DATE('{dt_str}', 'YYYY-MM-DD HH24:MI:SS')"
        )
        sql_surf = (
            f"select stacode, ddatetime, r from surf_cli_mul_hor_{year} "
            f"where length(stacode)=5 "
            f"and ddatetime = TO_DATE('{dt_str}', 'YYYY-MM-DD HH24:MI:SS')"
        )

        with engine.connect() as conn:
            awst_df = pd.read_sql(sql_awst, conn)
            surf_df = pd.read_sql(sql_surf, conn)

        precip_df = pd.concat([awst_df, surf_df], ignore_index=True)

        print(f"  Queried AWST and SURF data for {event['stacode']} at {event_time}")
        if precip_df.empty:
            print(f"  No precipitation records found for event {event['stacode']} at {event_time}")
            continue

        precip_df['stacode'] = precip_df['stacode'].astype(str).str.strip()
        precip_df['r'] = pd.to_numeric(precip_df['r'], errors='coerce')

        neighbor_codes = neighbor_map.get(event['stacode'], [])
        if not neighbor_codes:
            print(f"  Skipping event {event['stacode']} at {event_time}: no neighbors found")
            continue

        neighbor_df = precip_df[precip_df['stacode'].isin(neighbor_codes)].copy()
        if neighbor_df.empty:
            print(f"  Skipping event {event['stacode']} at {event_time}: no neighbor records in query result")
            continue

        neighbor_df = neighbor_df.merge(loc_df, on='stacode', how='inner')
        neighbor_df = neighbor_df.dropna(subset=['lon', 'lat', 'r'])
        if neighbor_df.empty:
            print(f"  Skipping event {event['stacode']} at {event_time}: neighbors have no lon/lat or r")
            continue

        print(f"Found {len(neighbor_df)} neighbor station record(s) for plotting")

        center_loc = loc_df[loc_df['stacode'] == event['stacode']]
        if center_loc.empty:
            print(f"Skipping event {event['stacode']} at {event_time}: missing center lon/lat")
            continue

        center_lon = float(center_loc.iloc[0].lon)
        center_lat = float(center_loc.iloc[0].lat)
        center_code = event['stacode']
        center_r = float(event['r'])
        if center_r == 338.8:
            print(f"  Warning: center station {center_code} has r=338.8, which may indicate a data issue")

        neighbor_gdf = gpd.GeoDataFrame(
            neighbor_df,
            geometry=gpd.points_from_xy(neighbor_df.lon, neighbor_df.lat),
            crs='EPSG:4326',
        )

        event_label = precip.iloc[idx].label
        if event_label == 'True': output_dir = OUTPUT_DIR_TRUE
        elif event_label == 'False': output_dir = OUTPUT_DIR_FALSE
        elif event_label == 'Uncertain': output_dir = OUTPUT_DIR_UNCERTAIN
        output_path = output_dir / f"{center_code}_{event_time.strftime('%Y%m%d_%H%M')}.png"
        plot_event(center_lon, center_lat, center_r, event_time, neighbor_gdf, output_path, center_code,event_label)
        print(f"Saved {output_path} (neighbors: {len(neighbor_gdf)})\n"
              "============================================================================")

    engine.dispose()
    print(f"Finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")


if __name__ == '__main__':
    main()
