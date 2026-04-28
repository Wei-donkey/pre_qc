# -*- coding: utf-8 -*-
"""
Plot precipitation events over 100mm against the data from neighboring stations as spatial scatter maps.
The script is organized in three major blocks:
1. target data loading,
2. neighboring data access from database,
3. plotting.
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
OUTPUT_DIR_FALSE = SRC_DIR.parent / 'figures' / 'false_precip_maps'
OUTPUT_DIR_UNCERTAIN = SRC_DIR.parent / 'figures' / 'uncertain_precip_maps'
OUTPUT_DIR_FALSE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_UNCERTAIN.mkdir(parents=True, exist_ok=True)

TARGET_PRECIP_FILE = SRC_DIR.parent / 'data' / 'gd_precip_hourly_awst_2003-2025.csv'
NEIGHBOR_FILE = SRC_DIR.parent / 'data' / 'gd_stations_neighbors.csv'
LOCATION_FILE = SRC_DIR.parent / 'data' / 'gd_stations_locations.csv'
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_TB_SURF = 'surf_cli_mul_hor'
DATA_TB_AWST = 'awst_cli_mul_hor'

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


def load_db_config(config_path: Path, section: str):
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8-sig')

    db = config[section]
    return {
        'user': db['user'],
        'password': db['password'],
        'host': db['host'],
        'port': db['port'],
        'service': db['service'],
    }


def set_plot_style():
    for key, value in STYLE_SETTINGS.items():
        if key == 'style':
            mpl.style.use(STYLE_SETTINGS['style'])
        else:
            mpl.rcParams[key] = value


def load_target_precip(target_precip_file: Path):
    df = pd.read_csv(target_precip_file, header=0, dtype=str)
    df = df.dropna(subset=['stacode', 'ddatetime', 'r']).copy()
    df['r'] = pd.to_numeric(df['r'])
    label_map = {0:'False',9:'Uncertain'}
    df['label'] = df['datatype'].astype(int).map(label_map)
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    df['year'] = df['ddatetime'].dt.year
    df['month'] = df['ddatetime'].dt.month
    df = df.sort_values(['ddatetime', 'r'], ascending=[True, False])
    df = df[df['r'] >= 100].reset_index(drop=True).copy()  # Filter for events with r >= 100mm
    return df


def load_station_neighbors(neighbor_file: Path):
    df = pd.read_csv(neighbor_file, header=0, dtype=str, quotechar='"', engine='python',
                     names=['stacode', 'neighbors', 'count'])
    df = df.dropna(subset=['stacode']).copy()
    df['stacode'] = df['stacode'].astype(str).str.strip()
    df['neighbors'] = df['neighbors'].fillna('').astype(str)
    df['neighbors_list'] = df['neighbors'].str.split(',').apply(lambda x: [code.strip() for code in x if code and code.strip()])
    df = df.set_index('stacode')['neighbors_list'].to_dict()

    return df


def load_sta_locations(location_file: Path):
    df = pd.read_csv(location_file, header=0, dtype=str, quotechar='"', engine='python',
                               names=['stacode', 'lon', 'lat', 'station_type'])
    df = df.dropna(subset=['lon', 'lat']).copy()
    df['stacode'] = df['stacode'].astype(str).str.strip()
    df['lon'] = pd.to_numeric(df['lon'])
    df['lat'] = pd.to_numeric(df['lat'])

    return df


def create_db_engine(db_config: dict[str, str]):
    password = quote(db_config['password'])
    conn_string = (
        f"oracle+oracledb://{db_config['user']}:{password}@{db_config['host']}"
        f":{db_config['port']}/{db_config['service']}"
    )
    return create_engine(conn_string, echo=False)


def fetch_neighbor_samples_from_db(engine, table_name_surf: str, table_name_awst: str,
                                   time_stt: str, time_end: str, neighbors_code: list[str], sta_location: pd.DataFrame):
    """
    Fetch neighbor station (neighbor_map) precipitation records 
    for a specific station (stacode) and time window (start_time_str - end_time_str).
    """
    
    strtime_stt, strtime_end = time_stt.strftime('%Y-%m-%d %H:%M:%S'), time_end.strftime('%Y-%m-%d %H:%M:%S')
    sql_surf = (
        f"select stacode, ddatetime, r from {table_name_surf} "
        f"where length(stacode)=5 "
        f"and ddatetime between TO_DATE('{strtime_stt}', 'YYYY-MM-DD HH24:MI:SS') "
        f"and TO_DATE('{strtime_end}', 'YYYY-MM-DD HH24:MI:SS') "
        f"and r<=184.4"
    )
    sql_awst = (
        f"select stacode, ddatetime, r from {table_name_awst} "
        f"where length(stacode)=5 "
        f"and ddatetime between TO_DATE('{strtime_stt}', 'YYYY-MM-DD HH24:MI:SS') "
        f"and TO_DATE('{strtime_end}', 'YYYY-MM-DD HH24:MI:SS') "
        f"and r<=184.4"
    )
    sql = f"{sql_surf} union all {sql_awst}"

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    
    df['stacode'] = df['stacode'].astype(str).str.strip()    
    df = df.set_index(['stacode'])
    df = df.loc[neighbors_code].reset_index()
    df['r'] = pd.to_numeric(df['r'])
    df = df.merge(sta_location[['stacode', 'lon', 'lat']], on='stacode', how='left')


def plot_precip_event(event_label: str, center_code: str, center_lon: float, center_lat: float, 
                      center_r: float, event_time: pd.Timestamp, neighbor_gdf: gpd.GeoDataFrame, output_path: Path,):
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)

    # Calculate extent based on furthest neighbor station from center, with gap
    if not neighbor_gdf.empty:
        distances = np.sqrt((neighbor_gdf.geometry.x - center_lon)**2 + (neighbor_gdf.geometry.y - center_lat)**2)
        max_distance = distances.max()
        gap = 0.02  # gap beyond furthest station
        extent_half = max_distance + gap
    else:
        extent_half = 0.46  # Default fallback
    
    extent = [center_lon - extent_half, center_lon + extent_half, center_lat - extent_half, center_lat + extent_half]
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
        # Scale scatter size based on precipitation (1 to 25)
        neighbor_gdf = neighbor_gdf.copy()
        neighbor_gdf['size'] = 10 + 15 * (neighbor_gdf['r'] / vmax)
        
        scatter = ax.scatter(
            neighbor_gdf.geometry.x,
            neighbor_gdf.geometry.y,
            c=neighbor_gdf['r'],
            cmap='viridis',
            vmin=0,
            vmax=vmax,
            s=neighbor_gdf['size'],
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
                row.geometry.y + 0.01,
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


def main():
    set_plot_style()

    target_precip = load_target_precip(TARGET_PRECIP_FILE)
    neighbor_map = load_station_neighbors(NEIGHBOR_FILE)
    sta_location = load_sta_locations(LOCATION_FILE)
    print(f"Loaded {len(target_precip)} event(s), {len(neighbor_map)} neighbor station entries, {len(sta_location)} station locations.")
    # target_precip = target_precip.copy()

    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)

    for idx, event in target_precip.iterrows():
        center_code = str(event['stacode']).strip()
        event_time = event['ddatetime']
        print(f"Processing event {idx+1}/{len(target_precip)}: {center_code} at {event_time}")

        center_loc = sta_location[sta_location['stacode'] == center_code]
        if center_loc.empty:
            print(f"Skipping event {event['stacode']} at {event_time}: missing center lon/lat")
            continue

        year = int(event['year'])
        data_tb_surf = f"{DATA_TB_SURF}_{year}" 
        data_tb_awst = f"{DATA_TB_AWST}_{year}"

        neighbors = neighbor_map.get(center_code, [])
        neighbors_code = sorted({center_code} | set(neighbors)) 

        neighbors_df = fetch_neighbor_samples_from_db(engine, data_tb_surf, data_tb_awst, event_time, event_time, neighbors_code, sta_location)

        center_lon = float(center_loc.iloc[0].lon)
        center_lat = float(center_loc.iloc[0].lat)
        center_r = float(event['r'])

        neighbor_gdf = gpd.GeoDataFrame(neighbors_df, 
                                        geometry=gpd.points_from_xy(neighbors_df.lon, neighbors_df.lat), 
                                        crs='EPSG:4326', )

        event_label = event['label']
        if event_label == 'False': output_dir = OUTPUT_DIR_FALSE
        elif event_label == 'Uncertain': output_dir = OUTPUT_DIR_UNCERTAIN
        output_path = output_dir / f"{center_code}_{event_time.strftime('%Y%m%d_%H%M')}.png"
        plot_precip_event(event_label, center_code, center_lon, center_lat, center_r, event_time, neighbor_gdf, output_path)
        print(f"Saved {output_path} (neighbors: {len(neighbor_gdf)})")

    engine.dispose()
    print(f"Finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")


if __name__ == '__main__':
    main()
