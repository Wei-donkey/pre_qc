# -*- coding: utf-8 -*-
"""
Created on Thu April 13 20:15:51 2026
This script queries awst-type target events (50<=r<=184.4) from Oracle database tables (awst_cli_mul_hor_yyyy),
finds neighbor stations from 'gd_stations_neighbors.csv', and extract precipitation records from Oracle 
(surf_cli_mul_hor_yyyy/ awst_cli_mul_hor_yyyy) for the target timestamp plus 'window_hours' before and after.
The default window is 2 hours, producing samples at [T-2, T-1, T, T+1, T+2].
Note1: Hourly precipitation between 50~184.4mm are to be inspected for their reliability.
Note2: Events are grouped into continuous time segments if their time intervals (Event Time +/- 2 hours) overlap, to optimize database querying.
       From each data segments, we extract neighboring samples for each center station of each event.
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
event_shreshold_min = 50.0
event_shreshold_max = 184.4

year_stt, year_end = 2003, 2003
years = range(year_stt, year_end+1)

OUTPUT = SRC_DIR.parent / 'data' / f"gd_event_uncertain_neigbhor_precip_{year_stt}-{year_end}.csv"


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
    """Load awst-type target events with 50<=r<= 184.4 from a specific year's datatable."""
    sql = (f"select stacode, ddatetime, r "
           f"from {table_name} "
           f"where r>={event_shreshold_min} "
           f"and r<={event_shreshold_max} ")
    
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


def rank_events_time_intervals(events: pd.DataFrame):
    # Rank/Sort events based on ddatetime
    events = events.sort_values(by='ddatetime').reset_index(drop=True)

    # Create time intervals for each event (Event Time +/- 2 hours)
    time_intervals = []
    events_list = [] # Store event info to map back later if needed, though we filter by time/stacode

    for _, event in events.iterrows():
        event_time = event['ddatetime']                
        time_stt = event_time - timedelta(hours=WINDOW_HOURS)
        time_end = event_time + timedelta(hours=WINDOW_HOURS)
        time_intervals.append((time_stt, time_end))
        
        # Keep track of original events for validation/extraction logic
        events_list.append({'stacode': str(event['stacode']).strip(),
                                'ddatetime': event_time,'r': event['r']})

    return events_list, time_intervals 

def merge_time_segments(intervals: list[tuple[datetime, datetime]]):
    """
    Merge overlapping or contiguous time intervals into continuous segments.
    Input: List of (start, end) tuples.
    Output: List of merged (start, end) tuples.
    """
    
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        
        # If current interval overlaps or is contiguous with the last one
        if current_start <= last_end:
            # Extend the end time if necessary
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
            
    return merged


def extract_segment_samples_from_db(engine, table_name_surf: str, table_name_awst: str,
                                   time_stt: datetime, time_end: datetime, 
                                   #neighbors_code: list[str]
                                   ):
    """
    Fetch precipitation records 
    for a specific time window (start_time_str - end_time_str).
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
    df['statype'] = 'N'

    return df


def extract_neighbor_samples_from_segment(seg_df: pd.DataFrame, event: dict, neighbor_map: dict[str, list[str]], 
                                          seg_stt: datetime, seg_end: datetime):

    station_code = event['stacode']
    event_time = event['ddatetime']

    if event_time > seg_end:
        return None, True
    if event_time < seg_stt:
        return None, False

    event_stt = event_time - timedelta(hours=WINDOW_HOURS)
    event_end = event_time + timedelta(hours=WINDOW_HOURS)
    neighbors = neighbor_map.get(station_code, [])

    # Check if this event falls within the current segment
    neighbor_mask = (seg_df['stacode'].isin(neighbors) &
                    (seg_df['ddatetime'] >= event_stt) &
                    (seg_df['ddatetime'] <= event_end))

    neighbor_df = seg_df[neighbor_mask].copy()
    neighbor_df['statype'] = 'N'

    center_df = pd.DataFrame([{'stacode': station_code, 
                                'ddatetime': event_time, 
                                'r': event['r'], 'statype': 'C'}])
    
    event_df = pd.concat([center_df, neighbor_df], ignore_index=True)

    return event_df, False


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

        # 1. Load target events from awst stations first
        events = fetch_false_awst_events_from_db(engine, data_tb_awst)
        if events.empty:
            print(f"  No events found for {year}.")
            continue
        
        # 2. Filter events to keep only stations present in neighbor_map (land stations)
        events = events[events['stacode'].isin(neighbor_map.keys())].reset_index(drop=True)
        if events.empty:
            print(f"  No land-station events found for {year}.")
            continue
        
        # 3. Rank events by time and create time intervals (Event Time +/- 2 hours)
        events_list, time_intervals = rank_events_time_intervals(events)

        # 4. Merge overlapping intervals into continuous segments
        time_segments = merge_time_segments(time_intervals)
        print(f" {len(time_intervals)} time intervals merged into {len(time_segments)} continuous time segments for database querying")

        # Optimized Loop: Use a pointer to track progress in events_list
        event_ptr = 0

        # 5. Loop through segments and fetch data blocks
        for seg_idx, (seg_stt, seg_end) in enumerate(time_segments):
            print(f"    Processing segment {seg_idx+1}/{len(time_segments)}: {seg_stt} to {seg_end}")
            seg_df = extract_segment_samples_from_db(engine, data_tb_surf, data_tb_awst, seg_stt, seg_end)
            
            # 6. Extract specific events and their neighbors from the in-memory block
            for event in events_list[event_ptr:]:            
                event_df, flag_next_seg = extract_neighbor_samples_from_segment(seg_df, event, neighbor_map, 
                                                                                seg_stt, seg_end)
                if flag_next_seg:
                    break
                else:
                    year_data.append(event_df)
                    event_ptr += 1

        if year_data:
            year_sample = pd.concat(year_data, ignore_index=True)
            year_sample.drop_duplicates(subset=['stacode', 'ddatetime'], inplace=True)
            all_data.append(year_sample)
        else:
            print(f"  No event samples found for {year}.")

    samples = pd.concat(all_data, ignore_index=True)
    # Ensure output directory exists
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(samples)} rows to {OUTPUT}:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    engine.dispose()


if __name__=='__main__':
    main()