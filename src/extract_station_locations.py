# -*- coding: utf-8 -*-
"""
Fetch longitude and latitude for stations in 广东 that have 5-digit codes and station type markers A/B.
Only stations matching v_prcode='广东', length(v01301)=5, and v02301 like '%A%' or '%B%' are returned.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from sqlalchemy import create_engine

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
TABLE_NAME = 't_othe_station_meta_basic_tab'
OUTPUT = SRC_DIR.parent / 'data' / 'gd_stations_locations.csv'


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


def fetch_locations_from_db(engine, table_name: str):
    sql = (
        f"select v01301 stacode, v05001 lat, v06001 lon, "
        f"case "
        f"when v02301 like '%A%' then 'surf' "
        f"when v02301 like '%B%' then 'awst' "
        f"end as statype "
        f"from {table_name} "
        f"where v_prcode = '广东' "
        f"and (v02301 like '%A%' or v02301 like '%B%') "
        f"and inland=1 "
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    return df


def main() -> None:
    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)

    print(f"Reading station metadata from table: {TABLE_NAME}")
    locations = fetch_locations_from_db(engine, TABLE_NAME)

    # Ensure output directory exists
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    locations.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"Wrote {len(locations)} rows to {OUTPUT}")

    engine.dispose()


if __name__ == '__main__':
    main()
