# -*- coding: utf-8 -*-
"""
Plot precipitation events of national weather stations over 50mm against 
the data from neighboring stations as spatial scatter maps and histograms. 
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import timedelta
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / 'data'
OUTPUT_DIR = SRC_DIR.parent / 'figures' / 'true_precip_map_hist'

LOCATION_FILE = DATA_DIR / 'gd_stations_locations.csv'
NEIGHBOR_FILE = DATA_DIR / 'gd_stations_neighbors.csv'
PRECIP_FILE = DATA_DIR / 'gd_event_true_neigbhor_precip_2003-2025.csv'

WINDOW_HOURS = 2
DPI = 300
FIGSIZE = (3, 3.5)

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


def set_plot_style():
    """Apply consistent plot styling."""
    for key, value in STYLE_SETTINGS.items():
        if key == 'style':
            mpl.style.use(STYLE_SETTINGS['style'])
        else:
            mpl.rcParams[key] = value


def load_data():
    """Load all necessary CSV files into memory."""
    print("Loading data files...")
    
    # Load station locations
    df_locs = pd.read_csv(LOCATION_FILE)
    print(f"  Station locations: {len(df_locs)} stations loaded")
    
    # Load neighbor lists
    df_neighbors_lst = pd.read_csv(NEIGHBOR_FILE)
    print(f"  Neighbor relationships: {len(df_neighbors_lst)} central stations loaded")
    
    # Load precipitation data
    df_precip = pd.read_csv(PRECIP_FILE)
    df_precip['ddatetime'] = pd.to_datetime(df_precip['ddatetime'])
    print(f"  Precipitation data: {len(df_precip)} records loaded")
    
    return df_locs, df_neighbors_lst, df_precip


def compute_max_precip_in_window(df_neighbor_data):
    """Compute maximum precipitation for each station within the time window.
    
    Args:
        df_neighbor_data: Filtered precipitation data within time window
        
    Returns:
        DataFrame with stacode and max precipitation value
    """
    # Group by station and compute maximum precipitation in the window
    df_max = df_neighbor_data.groupby('stacode')['r'].max().reset_index()
    df_max.columns = ['stacode', 'max_r']
    
    return df_max


def get_neighbors(central_code, df_neighbors_lst):
    """Get list of neighbor station codes for a given central station.
    
    Args:
        central_code: Station code of the central station
        df_neighbors_lst: DataFrame containing neighbor relationships
        
    Returns:
        List of neighbor station codes
    """
    mask = df_neighbors_lst['stacode'] == central_code
    if not mask.any():
        return []
    
    neighbors_str = df_neighbors_lst.loc[mask, 'neighbors'].values[0]
    if pd.isna(neighbors_str) or neighbors_str == '':
        return []
    
    # Parse comma-separated neighbor list
    neighbors = [n.strip() for n in str(neighbors_str).split(',')]
    return neighbors


def plot_event(central_code: str, 
               event_time: pd.Timestamp, 
               central_precip: float,
               df_precip: pd.DataFrame, 
               df_locs: pd.DataFrame, 
               df_neighbors_lst: pd.DataFrame, 
               output_dir: Path):
    """
    Plot scatter map and histogram for a specific precipitation event.
    """
    # 1. Determine time window
    time_stt = event_time - timedelta(hours=WINDOW_HOURS)
    time_end = event_time + timedelta(hours=WINDOW_HOURS)
    
    # 2. Get neighboring stations
    neighbor_codes = get_neighbors(central_code, df_neighbors_lst)
    
    # 3. Filter precipitation data by time window and stations
    mask_time = (df_precip['ddatetime'] >= time_stt) & (df_precip['ddatetime'] <= time_end)
    mask_stations = df_precip['stacode'].isin(neighbor_codes)
    df_neighbor_data = df_precip[mask_time & mask_stations].copy()
    
    if df_neighbor_data.empty:
        print(f"  No neighbor data found for central station {central_code} at {event_time}")
        return
    
    # 4. Compute maximum precipitation for each station in the window
    df_neighbor_max = compute_max_precip_in_window(df_neighbor_data)
    
    # 5. Merge with location data
    df_neighbor_max_locs = pd.merge(
        df_neighbor_max,
        df_locs[['stacode', 'lat', 'lon']],
        on='stacode',
        how='left'
    )
    
    # Drop rows without coordinates
    df_neighbor_max_locs = df_neighbor_max_locs.dropna(subset=['lat', 'lon'])
    
    if df_neighbor_max_locs.empty:
        print(f"  No location data for stations in event {central_code} at {event_time}")
        return
    
    # 6. Create GeoDataFrame for mapping (exclude central station)
    gdf_neighbor = gpd.GeoDataFrame(
        df_neighbor_max_locs,
        geometry=gpd.points_from_xy(df_neighbor_max_locs.lon, df_neighbor_max_locs.lat),
        crs='EPSG:4326'
    )
    
    # Get central station location
    central_loc = df_locs[df_locs['stacode'] == central_code]
    if central_loc.empty:
        print(f"  Central station {central_code} not found in location data")
        return
    
    central_lon = float(central_loc.iloc[0].lon)
    central_lat = float(central_loc.iloc[0].lat)
    
    # 7. Calculate extent based on furthest neighbor from center
    if not gdf_neighbor.empty:
        distances = np.sqrt(
            (gdf_neighbor.geometry.x - central_lon)**2 + 
            (gdf_neighbor.geometry.y - central_lat)**2
        )
        max_distance = distances.max()
        gap = 0.05  # Gap beyond furthest station
        extent_half = max_distance + gap
    else:
        extent_half = 0.3  # Default fallback
    
    # Ensure central station is centered
    extent = [
        central_lon - extent_half,
        central_lon + extent_half,
        central_lat - extent_half,
        central_lat + extent_half
    ]
    
    # 8. Create figure with ultra-tight margins and 4:1 height ratio
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.14)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 0.8], hspace=0.14)
    
    # Upper subplot: Map with cartopy projection
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    
    # Set extent
    ax_map.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add geographic features
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                        edgecolor='face', facecolor='#f0f0f0')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', 
                                         edgecolor='face', facecolor='#e6f0ff')
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', 
                                             edgecolor='black', facecolor='none')
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', 
                                           edgecolor='black', facecolor='none')
    
    ax_map.add_feature(land, zorder=0)
    ax_map.add_feature(ocean, zorder=0)
    ax_map.add_feature(coastline, linewidth=0.5, zorder=1)
    ax_map.add_feature(borders, linewidth=0.5, alpha=0.5, zorder=1)
    
    # Add gridlines with one decimal precision and fixed 0.1 degree interval
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color='gray', 
                          alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 5, 'rotation': 20}  # Rotate labels to avoid overlapping
    gl.ylabel_style = {'size': 5}
    
    # Set gridline positions with fixed 0.1 degree interval
    x_start = np.ceil(extent[0] * 10) / 10
    y_start = np.ceil(extent[2] * 10) / 10
    x_end = np.floor(extent[1] * 10) / 10
    y_end = np.floor(extent[3] * 10) / 10
    
    gl.xlocator = mticker.FixedLocator(np.arange(x_start, x_end + 0.1, 0.1))
    gl.ylocator = mticker.FixedLocator(np.arange(y_start, y_end + 0.1, 0.1))
    
    # Format latitude labels with 1 decimal and N/S suffix
    def lat_formatter(val, pos):
        if val >= 0:
            return f'{val:.1f}°N'
        else:
            return f'{abs(val):.1f}°S'
    
    gl.yformatter = mticker.FuncFormatter(lat_formatter)
    
    # Plot neighbor stations
    if not gdf_neighbor.empty:
        vmax = float(max(gdf_neighbor['max_r'].max(), 0.1))
        
        # Scale scatter size based on precipitation
        gdf_neighbor = gdf_neighbor.copy()
        gdf_neighbor['size'] = 10 + 15 * (gdf_neighbor['max_r'] / vmax)
        
        scatter = ax_map.scatter(
            gdf_neighbor.geometry.x,
            gdf_neighbor.geometry.y,
            c=gdf_neighbor['max_r'],
            cmap='Blues',
            vmin=0,
            vmax=np.ceil(vmax),  # Ceiling for integer colorbar
            s=gdf_neighbor['size'],
            edgecolor='black',
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        
        # Colorbar with integer labels, stretched to match map height
        cbar = fig.colorbar(scatter, ax=ax_map, shrink=0.9, pad=0.02, aspect=20)
        cbar.set_label('Precipitation (mm)') #, fontsize=6
        cbar.ax.tick_params()#labelsize=4
        cbar.locator = mticker.MaxNLocator(integer=True)
        cbar.update_ticks()
        
        # Label large precipitation values with background
        large_values = gdf_neighbor[gdf_neighbor['max_r'] >= 5]
        for _, row in large_values.iterrows():
            ax_map.text(
                row.geometry.x + 0.02,
                row.geometry.y,
                f"{row['max_r']:.1f}",
                transform=ccrs.PlateCarree(),
                # fontsize=6,
                color='darkblue',
                va='center',
                ha='left',
                zorder=3,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3),
            )
    
    # Plot central station
    ax_map.scatter(
        central_lon,
        central_lat,
        marker='*',
        c='red',
        s=80,
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
        zorder=3,
        label='Central',
    )
    
    # Label central station value with background
    ax_map.text(
        central_lon + 0.03,
        central_lat,
        f"{central_precip:.1f}",
        transform=ccrs.PlateCarree(),
        # fontsize=8,
        color='red',
        va='center',
        ha='left',
        zorder=4,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3),
    )
    
    # Add small annotation with station code and event time (bottom-left corner)
    ax_map.text(
        0.5, 0.93,
        f'{central_code}@ {event_time.strftime("%Y-%m-%d %H:%M")}',
        transform=ax_map.transAxes,
        fontsize=6,  # Match the star marker size on map
        color='black',
        va='bottom',
        ha='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=None, linewidth=0.5, pad=2),
        zorder=5,
    )
    
    # Title annotation removed to maximize map area - info in filename only
    
    # Lower subplot: Histogram (include central station value)
    ax_hist = fig.add_subplot(gs[1])
    precip_values = pd.concat([gdf_neighbor['max_r'], pd.Series([central_precip])]).dropna()
    
    if not precip_values.empty:
        ax_hist.hist(precip_values, bins=15, color='steelblue', 
                     edgecolor='black', linewidth=0.5)
        ax_hist.set_xlim(left=0)  # Fix starting point at 0
        
        # Add red vertical line at central station value
        ax_hist.axvline(x=central_precip, color='red', linestyle='--', linewidth=1, alpha=0.8)
        
        # Add label beside the vertical line
        ax_hist.text(
            central_precip + 0.5,  # Offset to the right of the line
            ax_hist.get_ylim()[1] * 0.95,  # Near top of y-axis
            f"{central_precip:.1f}",
            color='red',
            fontsize=6,
            va='top',
            ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3)
        )
        
        ax_hist.set_xlabel('Precipitation (mm)',labelpad=2)#, fontsize=6
        ax_hist.set_ylabel('Frequency')#, fontsize=6
        
        # Set y-axis with integer labels and dynamic interval
        max_freq = len(precip_values)  # Maximum possible frequency
        if max_freq >= 8:
            ax_hist.yaxis.set_major_locator(mticker.MultipleLocator(2))  # Interval of 2 if max ≤ 8
        else:
            ax_hist.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Auto integer locator
        
        ax_hist.tick_params()#labelsize=4
    else:
        ax_hist.text(
            0.5, 0.5, 'No Data',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax_hist.transAxes,
            # fontsize=6
        )
    
    # Save figure
    safe_time = event_time.strftime("%Y%m%d_%H%M")
    filename = f"event_{central_code}_{safe_time}.png"
    filepath = output_dir / filename
    
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def main():
    # Apply plot style
    set_plot_style()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data: locations, neighbors list, and precipitation data
    df_locs, df_neighbors_lst, df_precip = load_data()
    
    # Filter for central events (statype == 'C')
    print("\nFiltering central events (statype='C')...")
    df_central = df_precip[df_precip['statype'] == 'C'].copy()
    print(f"  Found {len(df_central)} central event records")
    
    # Sort chronologically as per project specification
    df_central = df_central.sort_values('ddatetime').reset_index(drop=True)
    print(f"  Unique central events: {len(df_central)}")
    
    # Process each event
    for idx, row in enumerate(df_central[:10].itertuples(), 1):
        central_code = row.stacode
        central_precip = row.r
        event_time = row.ddatetime
        
        print(f"[{idx}/{len(df_central)}] Processing event: {central_code} at {event_time}")
        
        try:
            plot_event(
                central_code,
                event_time,
                central_precip,
                df_precip,
                df_locs,
                df_neighbors_lst,
                OUTPUT_DIR
            )
        except Exception as e:
            print(f"  ERROR: Failed to process event {central_code} at {event_time}: {e}")
            continue
    
    print(f"\nCompleted! Plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
