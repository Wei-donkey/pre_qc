# -*- coding: utf-8 -*-
"""
Created on Thu April 13 20:15:51 2026
Query two datasets and combine they before output: 
1. Hourly precipitation over 25mm from national observatories, all of which are absolutely correct thanks to manual inspection;
2. Hourly precipitation over 184.4mm from automatic weather stations, which are absolutely incorrect beyond climate limitations.
Concatenate them into one file and ascend all records in time order.
@author: Wei Liu
"""

from urllib.parse import quote
import os
import pandas as pd
import configparser

src_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(src_dir)
data_dir = proj_dir + '\\data'

# ========== LOAD CONFIGURATION ==========
config = configparser.ConfigParser()
config.read(src_dir + '\\config_db.ini', encoding='utf-8-sig')

# Database settings
DB_CONFIG = {
    'username': config['CROSS_WEATHER']['user'],
    'password': config['CROSS_WEATHER']['password'],
    'host': config['CROSS_WEATHER']['host'],
    'port': config['CROSS_WEATHER']['port'],
    'service': config['CROSS_WEATHER']['service']
}

statypes = ['surf','awst']
data_tb = 'cli_mul_hor'
stainfo_tb = 't_othe_station_meta_basic_tab'
prov = '广东'

year_stt, year_end = 2003, 2025
years = range(year_stt, year_end+1)

output_path = data_dir + f"\\gd_precip_hourly_{year_stt}-{year_end}.csv"
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
        host_db = DB_CONFIG['host']
        port_db = DB_CONFIG['port']
        service_db = DB_CONFIG['service']
    except NameError:
        print("Global configuration variables are not defined.")
        return None
    except KeyError as e:
        print(f"Missing key in DB_CONFIG: {e}")
        return None

    # Encode password for URL
    pwd_db = quote(pwd_db)
    
    engine = None    
    try:

        from sqlalchemy import create_engine
        # create engine for reading data from oracle database
        conn_string = f"oracle+oracledb://{user_db}:{pwd_db}@{host_db}:{port_db}/{service_db}"
        engine = create_engine(conn_string, echo=False)        

        all_data = []
        for year in years:
            year_data = []
            
            for statype in statypes:
                data_tb_year = f'{statype}_{data_tb}_{year}'
                            
                base_sql = (
                    f"select stacode,ddatetime,r "
                    f"from {data_tb_year} "
                    f"where stacode in ("
                    f"select v01301 from {stainfo_tb} " 
                    f"where v_prcode ='{prov}'"
                )
                if statype == 'surf':
                    sql = base_sql + " and v02301 like \'%A%\') and r >= 25"
                elif statype == 'awst':
                    sql = base_sql + " and v02301 like \'%B%\') and r > 184.4"
                
                df = db_reader(engine, sql)
            
                if df is None:
                    print(f"No {statype} data for {year}")
                    continue
                else:
                    print(f"Extracted {statype} data for {year}")
                    if statype == 'surf': 
                        df['statype'], df['label'] = 'surf', 1
                    else:
                        df['statype'], df['label'] = 'awst', 0
                    year_data.append(df)
            
            if year_data:
                combined_year = pd.concat(year_data, ignore_index=True)
                combined_year = combined_year.sort_values(by=['ddatetime', 'stacode']) 
                all_data.append(combined_year)
            
        if all_data:
            final_combined = pd.concat(all_data, ignore_index=True)
            final_combined.to_csv(output_path, index=False)
            print('Data saved to CSV file.')
        else:
            print('No data returned.')
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if engine is not None:
            engine.dispose()

if __name__=='__main__':
    main()