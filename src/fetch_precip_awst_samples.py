# -*- coding: utf-8 -*-
"""
Created on Thu April 13 20:15:51 2026
Query two datasets from automatic weather stations: 
1. Hourly precipitation over 184.4mm which are absolutely incorrect beyond climate limitations.
2. Hourly precipitation between 50~184.4mm which are to be inspected for their reliability.
The two datasets are merged together and sorted by time and station code.
Concatenate them into one file and ascend all records in time order.
@author: Wei Liu
"""

from urllib.parse import quote
import os
import pandas as pd
import configparser
from datetime import datetime

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = PROJ_DIR + '\\data'

DATA_TB = 'cli_mul_hor'
PROV = '广东'

# ========== LOAD CONFIGURATION ==========
config = configparser.ConfigParser()
config.read(SRC_DIR + '\\config_db.ini', encoding='utf-8-sig')

# Database settings
DB_CONFIG = {
    'username': config['CROSS_WEATHER']['user'],
    'password': config['CROSS_WEATHER']['password'],
    'host': config['CROSS_WEATHER']['host'],
    'port': config['CROSS_WEATHER']['port'],
    'service': config['CROSS_WEATHER']['service']
}

datatypes = ['false','uncertain']
statypes = ['awst','awst']

year_stt, year_end = 2003, 2025
years = range(year_stt, year_end+1)

OUTPUT_PATH = DATA_DIR + f"\\gd_precip_hourly_{year_stt}-{year_end}.csv"
def db_reader(engine, sql):
    with engine.connect() as con:    
        df = pd.read_sql(sql, engine)
    if df.empty:
        return None
    else:
        return df
    
def main():
    try:
        user_db = DB_CONFIG['username']
        pwd_db = DB_CONFIG['password']
        # Encode password for URL
        pwd_db = quote(pwd_db)
        host_db = DB_CONFIG['host']
        port_db = DB_CONFIG['port']
        service_db = DB_CONFIG['service']
    except NameError:
        print("Global configuration variables are not defined.")
        return None
    except KeyError as e:
        print(f"Missing key in DB_CONFIG: {e}")
        return None
    
    engine = None    
    try:

        from sqlalchemy import create_engine
        # create engine for reading data from oracle database
        conn_string = f"oracle+oracledb://{user_db}:{pwd_db}@{host_db}:{port_db}/{service_db}"
        engine = create_engine(conn_string, echo=False)        

        all_data = []
        for year in years:
            year_data = []
            
            for datatype in datatypes:
                statype = statypes[datatypes.index(datatype)]
                data_tb_year = f"{statype}_{DATA_TB}_{year}"
                            
                base_sql = (
                    f"select stacode,ddatetime,r "
                    f"from {data_tb_year} "
                    f"where length(stacode)=5" 
                )
                if datatype == 'false':
                    sql = base_sql + " and r > 184.4"
                elif datatype == 'uncertain':
                    sql = base_sql + " and r <= 184.4 and r >= 50"
                
                print(f"extracting {statype} {datatype} data for {year}")
                df = db_reader(engine, sql)
            
                if df is None:
                    print(f"no {datatype} data for {year}")
                    continue
                else:
                    if datatype == 'false':
                        df['statype'], df['datatype'] = 'awst', 1
                    elif datatype == 'uncertain':
                        df['statype'], df['datatype'] = 'awst', 9
                    year_data.append(df)
            
            if year_data:
                combined_year = pd.concat(year_data, ignore_index=True)
                combined_year = combined_year.sort_values(by=['ddatetime', 'stacode']) 
                all_data.append(combined_year)
            
        if all_data:
            final_combined = pd.concat(all_data, ignore_index=True)
            final_combined.to_csv(OUTPUT_PATH, index=False)
            print(f"Data saved to CSV file.")
        else:
            print('No data returned.')
        print(f"Finished:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if engine is not None:
            engine.dispose()

if __name__=='__main__':
    main()