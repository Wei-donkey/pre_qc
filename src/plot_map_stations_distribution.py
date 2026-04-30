# -*- coding: utf-8 -*-
"""
Plot the distribution of meteorological stations in Guangdong Province.
Stations are categorized by type: surf-type and awst-type.
"""

from __future__ import annotations

import matplotlib as mpl

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / 'data'
OUTPUT_DIR = SRC_DIR.parent / 'figures' / 'station_maps'

STATION_FILE = DATA_DIR / 'gd_stations_locations.csv'
SHP_FILE = DATA_DIR / 'external' / 'CHN_Province_Border.shp'
OUTPUT_FILE = OUTPUT_DIR / 'gd_stations_distribution_map.png'

# Geographic bounds for Guangdong Province
LON_MIN, LON_MAX = 109.0, 119.0
LAT_MIN, LAT_MAX = 19.0, 26.0

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


def load_station_data(station_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load station data and separate by station type.
    
    Returns:
        Tuple of (surf_stations, awst_stations) DataFrames
    """
    df = pd.read_csv(station_file, encoding='utf-8-sig')
    
    # Separate by station type
    surf_stations = df[df['statype'] == 'surf'].copy()
    awst_stations = df[df['statype'] == 'awst'].copy()
    
    return surf_stations, awst_stations


def plot_stations_map(surf_stations: pd.DataFrame, awst_stations: pd.DataFrame):
    """
    Plot station distribution map using cartopy for professional geographic visualization.
    """
    # Create figure with cartopy projection
    fig, ax_map = plt.subplots(
        figsize=(14, 10),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    
    # Set map extent to Guangdong Province
    ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    # Add geographic features
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                        edgecolor='face', facecolor='#f0f0f0')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', 
                                         edgecolor='face', facecolor="#73a1e7")
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', 
                                             edgecolor='black', facecolor='none')
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', 
                                           edgecolor='black', facecolor='none')
    
    # ax_map.add_feature(land, zorder=0)
    ax_map.add_feature(ocean, zorder=0)
    # ax_map.add_feature(coastline, linewidth=0.5, zorder=1)
    # ax_map.add_feature(borders, linewidth=0.5, alpha=0.5, zorder=1)

     # load Guangdong boundary from a shapefile
    gd_boundary = gpd.read_file(SHP_FILE)  # Replace with your file path
    ax_map.add_geometries(gd_boundary.geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=0.5, alpha=1)


    # Add gridlines
    gl = ax_map.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.7,
        linestyle='-',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'rotation': 0}
    gl.ylabel_style = {'size': 10, 'rotation': 0}

    gl.xlocator = mticker.FixedLocator(np.arange(LON_MIN, LON_MAX + 1, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(LAT_MIN, LAT_MAX + 1, 1))

    # Plot surf-type stations (blue circles)
    if not surf_stations.empty:
        ax_map.scatter(
            surf_stations['lon'],
            surf_stations['lat'],
            transform=ccrs.PlateCarree(),
            c='blue',
            marker='o',
            s=20,
            alpha=1,
            label='National Observatory Station',
            zorder=6,
        )
    
    # Plot awst-type stations (red triangles)
    if not awst_stations.empty:
        ax_map.scatter(
            awst_stations['lon'],
            awst_stations['lat'],
            transform=ccrs.PlateCarree(),
            c='red',
            edgecolor='white',
            linewidths=0.2,
            marker='^',
            s=30,
            alpha=0.9,
            label='Automatic Weather Station',
            zorder=5,
        )
    
    # Add legend in bottom right corner
    legend = ax_map.legend(
        loc='lower left',
        frameon=True,
        framealpha=0.9,
        fontsize=11,
    )
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)


    # --- Add North Arrow ---
    # Position in axes coordinates (0-1)
    arrow_x, arrow_y = 0.95, 0.95 
    arrow_length = 0.04
    
    # Draw arrow using annotate
    ax_map.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                    xytext=(arrow_x, arrow_y - arrow_length), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    
    # Add 'N' label
    ax_map.text(arrow_x, arrow_y + 0.01, 'N', transform=ax_map.transAxes,
                ha='center', va='bottom', fontsize=12, fontweight='bold')


    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
    )
    print(f"Map saved to: {OUTPUT_FILE}")
    
    # Display
    # plt.show()


def main() -> None:
    print(f"Loading station data from: {STATION_FILE}")
    surf_stations, awst_stations = load_station_data(STATION_FILE)
    
    print(f"Surf-type stations: {len(surf_stations)}")
    print(f"AWST-type stations: {len(awst_stations)}")
    print(f"Total stations: {len(surf_stations) + len(awst_stations)}")
    
    plot_stations_map(surf_stations, awst_stations)


if __name__ == '__main__':
    main()