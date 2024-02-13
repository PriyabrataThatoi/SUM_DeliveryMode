# read clean data
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
import numpy as np

df = pd.read_csv('resdf_delivery_mode_2024-02-10.csv')


df['prvs_lscs'] = df['indication'].str.contains('LSCS', case=False, na=False).astype(int)
df['prvs_abortion'] = df['indication'].str.contains('ABORTION', case=False, na=False).astype(int)
df['oligo'] = df['indication'].str.contains('OLIGO', case=False, na=False).astype(int)
df['meconium'] = df['indication'].str.contains('MECONIUM', case=False, na=False).astype(int)
df['iugr'] = df['indication'].str.contains('IUGR', case=False, na=False).astype(int)
df['gdm'] = df['indication'].str.contains('GDM', case=False, na=False).astype(int)
df['hyperthyraidisim'] = df['diagnosis'].str.contains('HYPOTHYRAIDISIM', case=False, na=False).astype(int)

def extract_info(df, column_name, start_phrase, end_phrase=None):
    if end_phrase:
        # Extracting data between start_phrase and end_phrase
        df[column_name] = df['usg'].str.extract(f'{start_phrase}(.+?){end_phrase}')
    else:
        # Extracting data after start_phrase if no end_phrase is provided
        df[column_name] = df['usg'].str.extract(f'{start_phrase}(.+)')
    # Cleaning extracted data
    df[column_name] = df[column_name].str.strip()

def extract_efw(df):
    # Extracting only the first numeric part before the space
    df['E.F.W'] = df['usg'].str.extract('E.F.W- (\d+)').astype(str)
    
# Extracting each piece of information
extract_info(df, 'FHR', 'FHR-', '/MIN')
extract_info(df, 'Grade', 'GRADE-', ',')
extract_info(df, 'E.F.W', 'E.F.W-', ' GMS')
extract_info(df, 'A.F.I', 'A.F.I- ', ' CM')
extract_info(df, 'Placenta', 'PLACENTA- ', ',')
extract_info(df, 'Cervix_Length', 'CERVIX-', 'CM')
df['fetus_type'] = df['usg'].str.extract(r'(\b\w+)\s+LIVE')

df['fetus_type'] = np.where(df.fetus_type.isin(['SINGL', 'SINGLWE', 'WITH']),'SINGLE', df['fetus_type'])

extract_efw(df)

df = df[~df.usg.isna()]
