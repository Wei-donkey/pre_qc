# -*- coding: utf-8 -*-
"""
Created on Thu April 13 20:15:51 2026
This script queries awst-type target events (r >= 184.4mm) from Oracle database tables (awst_cli_mul_hor_yyyy),
finds neighbor stations from 'gd_stations_neighbors.csv', and extract precipitation records from Oracle 
(surf_cli_mul_hor_yyyy/ awst_cli_mul_hor_yyyy) for the target timestamp plus 'window_hours' before and after.
The default window is 2 hours, producing samples at [T-2, T-1, T, T+1, T+2].
Note: Hourly precipitation over climate limit-184.4mm are absolutely incorrect.
"""

from __future__ import annotations

import configparser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
import pandas as pd
from sqlalchemy import create_engine

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_TB_SURF = 'surf_cli_mul_hor'
DATA_TB_AWST = 'awst_cli_mul_hor'
NEIGHBOR_FILE = SRC_DIR.parent / 'data' / 'gd_stations_neighbors.csv'
WINDOW_HOURS = 2
event_shreshold = 184.4

year_stt, year_end = 2003, 2025
years = range(year_stt, year_end+1)

OUTPUT = SRC_DIR.parent / 'data' / f"gd_event_false_neigbhor_precip_{year_stt}-{year_end}.csv"


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


def create_db_engine(db_config: dict[str, str]):
    password = quote(db_config['password'])
    conn_string = (f"oracle+oracledb://{db_config['user']}:{password}@{db_config['host']}"
                   f":{db_config['port']}/{db_config['service']}")
    return create_engine(conn_string, echo=False)


def fetch_false_awst_events_from_db(engine, table_name: str):
    """Load awst-type target events with r > 184.4 from a specific year's datatable."""
    sql = (f"select stacode, ddatetime, r "
           f"from {table_name} "
           f"where r>{event_shreshold} ")
    
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    
    return df


def load_station_neighbors(neighbor_file: Path):
    df = pd.read_csv(neighbor_file, header=0, dtype=str, quotechar='"', engine='python')
    df['stacode'] = df['stacode'].astype(str).str.strip()
    df['neighbors'] = df['neighbors'].fillna('').astype(str)
    df['neighbors_list'] = df['neighbors'].str.split(',').apply(
        lambda x: [code.strip() for code in x if code and code.strip()])
    return df.set_index('stacode')['neighbors_list'].to_dict()


def extract_neighbor_samples_from_db(engine, table_name_surf: str, table_name_awst: str,
                                   time_stt: datetime, time_end: datetime, neighbors_code: list[str]):
    """
    Fetch neighbor station (neighbor_map) precipitation records 
    for a specific station (stacode) and time window (start_time_str - end_time_str).
    """    
    strtime_stt = time_stt.strftime('%Y-%m-%d %H:%M:%S')
    strtime_end = time_end.strftime('%Y-%m-%d %H:%M:%S')

    sql_surf = (
        f"select stacode, ddatetime, r from {table_name_surf} "
        f"where ddatetime between TO_DATE('{strtime_stt}', 'YYYY-MM-DD HH24:MI:SS') "
        f"and TO_DATE('{strtime_end}', 'YYYY-MM-DD HH24:MI:SS') "
        # f"and r >= 0.1 "
        f"and r <= 184.4 "
        # f"and length(stacode)=5 "
    )
    sql_awst = (
        f"select stacode, ddatetime, r from {table_name_awst} "
        f"where ddatetime between TO_DATE('{strtime_stt}', 'YYYY-MM-DD HH24:MI:SS') "
        f"and TO_DATE('{strtime_end}', 'YYYY-MM-DD HH24:MI:SS') "
        # f"and r >= 0.1 "
        f"and r <= 184.4 "
        # f"and length(stacode)=5 "
    )
    sql = f"{sql_surf} union all {sql_awst}"

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    
    df['stacode'] = df['stacode'].astype(str).str.strip()
    df = df.set_index('stacode')
    df_filter = df[df.index.isin(neighbors_code)].reset_index()
    df_filter['statype'] = 'N'

    return df_filter


def main():
    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)

    neighbor_map = load_station_neighbors(NEIGHBOR_FILE)

    all_data = []
    for year in years:
        year_data = []
        print(f'\n{"="*60}')        
        print(f"extracting precipitation samples from {year}")

        data_tb_surf = f"{DATA_TB_SURF}_{year}"
        data_tb_awst = f"{DATA_TB_AWST}_{year}"

        # Load target events from awst stations first
        events = fetch_false_awst_events_from_db(engine, data_tb_awst)
        if events.empty:
            print(f"  No events found for {year}.")
            continue
        
        # Filter events to keep only stations present in neighbor_map (land stations)
        events = events[events['stacode'].isin(neighbor_map.keys())].reset_index(drop=True)
        if events.empty:
            print(f"  No land-station events found for {year}.")
            continue
        
        for idx, event in events.iterrows():
            print(f"  Processing event {idx+1}/{len(events)}: {event['stacode']} at {event['ddatetime']}")
            station_code = str(event['stacode']).strip()

            neighbors = neighbor_map.get(station_code, [])
            # neighbors_code = sorted({station_code} | set(neighbors))

            event_time = event['ddatetime']
            time_stt = event_time - timedelta(hours=WINDOW_HOURS)
            time_end = event_time + timedelta(hours=WINDOW_HOURS)

            neighbor_df = extract_neighbor_samples_from_db(engine, data_tb_surf, data_tb_awst, 
                                                          time_stt, time_end, neighbors)
            center_df = pd.DataFrame([{'stacode': station_code, 'ddatetime': event_time, 'r': event['r'], 'statype': 'C'}])
            event_df = pd.concat([center_df, neighbor_df], ignore_index=True)

            year_data.append(event_df)

        year_sample = pd.concat(year_data, ignore_index=True)
        year_sample.drop_duplicates(subset=['stacode', 'ddatetime'], inplace=True)
        all_data.append(year_sample)

    samples = pd.concat(all_data, ignore_index=True)
    # Ensure output directory exists
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(samples)} rows to {OUTPUT}:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    engine.dispose()


if __name__=='__main__':
    main()