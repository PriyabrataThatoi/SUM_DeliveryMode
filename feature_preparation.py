# read clean data
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
import numpy as np
import random
import re

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

## To calculate the upper and lower range for imputation
def cal_lower_upper(df,min_range,max_range):
    # Generate random numbers and fill null values
    random_values = [random.uniform(min_range, max_range) if pd.isna(val) else val for val in df['kg']]
    
    return random_values

## Replace age and drop age with 16years

df['age'] = df['age'].replace({269.0: 26,257:25,16:21})



## Weight#############
## Replace missing weight with +/- in std deviation for weight and height
mean_weight = df['kg'].mean()
std_dev_weight=df['kg'].std()
# Define the range for random numbers
kg_upper_range=mean_weight+std_dev_weight
kg_lower_range=mean_weight-std_dev_weight
df['kg_upd']=cal_lower_upper(df,kg_lower_range,kg_upper_range)


##### Height ###########
## Replace missing weight with +/- in std deviation for weight and height
mean_height = df['height'].mean()
std_dev_height=df['height'].std()
# Define the range for random numbers
height_upper_range=mean_height+std_dev_height
height_lower_range=mean_height-std_dev_height
df['height_upd']=cal_lower_upper(df,height_lower_range,height_upper_range)

## Convert height from cms to metres
df['height_upd']=df['height_upd']/100



## Replace missing haemoglobin with mean
df['hb']=df['hb'].replace({'11..9':'11.9','10..2':'10.2','10 .1':'10.1'})
df['hb']=df['hb'].astype(float)
mean_hb = df['hb'].mean()
df['hb'].fillna(mean_hb, inplace=True)

## Replace missing ga_weeks with mean
df.ga_weeks.fillna(round(df[~df.ga_weeks.isna()].ga_weeks.astype(int).mean()))


##Looking into age and can be imputed ##TBD
#Replace for bp
df['bp']=df['bp'].replace({'10.2':'110/70'})
df['bp'].fillna('110/70', inplace=True)


## Drop rows who opted for LSCS option by default
df['maternal_request'] = df['indication'].str.contains('MATERNAL', case=False, na=False).astype(int)
df=df[df['maternal_request']==0]

df['hhh'].replace({'110/70':'NO- REACTIVE'},inplace=True)


df.fetus_type = df.fetus_type.fillna('SINGLE')


df['bmi']=((df['kg_upd'])/(df['height_upd']*df['height_upd']))

df['Placenta']=df['Placenta'].astype(str)


###Categories merged
# FUNDAL: FUNDAL FUNDAL LEFT ANTERIOR FUNDAL-POSTERIOR FUNDAL-POSTERIOR .' FUNDAL-ANTERIOR FUNDAL ANTERIOR FUNDAL . ' FUNDAL . '

# FUNDO-ANTERIOR:FUNDO-ANTERIOR FUNDO-ANTERIOR . ' FUNDO-ANTERIOR .' FUNDO-ANTERIOR . CERVIX-3.2 CM FUNDO- ANTERIOR . ' FUNDO- ANTERIOR FUNDOANT IN LOCATION .' FUNDO-ANTERIOR RIGHT LATERAL EXTENSION. NOT LOW LYING .' FUNDO-ANTERIOR NOT LOW LYING

# POSTERIOR:POSTERIOR POSTERIOR NOT LOW LYING POSTERIOR FUNDAL . A.F.I- 11.3 CM POSTERIOR RTLATERAL POSTERIOR LOW LYING POSTERIOR . ' POSTERIOR . (USG IN L.R )' POSTERIOR LEFT LATERAL POSTERIOR & FUNDAL POSTERIOR-FUNDAL . '

# ANTERIOR:ANTERIOR ANTERIOR . ANTERIOR . ' ANTERIOR . A.F.I- 10 CM ANTERIOR LEFT LATERAL ANTERIOR RIGHT FUNDAL ANTERIOR FUNDAL ANTERIOR FUNDAL . GARDE-3 . ' ANTERIOR TO THE RIGHT NOT LOW LYING . ' ANTERIOR AND LEFT LATERAL ANTERIOR LEFT LATERAL . ' ANTERIOR . '

# NAN:nan

# LEFT:LEFT LEFT LATERAL LEFT FUNDAL

# RIGHT:RIGHT RIGHT ANTERIOR RIGHT LATERAL RIGHT LATERAL POSTERIOR (SINGLE)- RIGHT LATERAL '

# OTHERS:NOT LOW LYINY POSTERIOR NATERIOR FUNDIC FUNDO-POSTERIOR FUNDO-POSTERIOR . ' FUNDO-'' FUNDO- POSTERIOR FUNDIC POSTERIOR FUNDIC PARTLY ANTERIOR FUNDIC RIGHT LATERAL FUNDIC-PARTLY ANTERIOR PARTLY' FUNDP-POSTERIOR SEPARATION LATERAL FUNDO POSTERIOR WITH LEFT LATERAL LATERLAL POSTERIOR FUN-ANTERIOR POSTEROLATERALLY


## Code to merge categories in placenta and perform one hot encoding
mapping = {
    'FUNDAL': 'FUNDAL',
    'FUNDO-ANTERIOR': 'FUNDO-ANTERIOR',
    'POSTERIOR': 'POSTERIOR',
    'ANTERIOR': 'ANTERIOR',
    'nan': 'nan',
    'LEFT': 'LEFT',
    'RIGHT': 'RIGHT'
}

# Create new columns based on mapping
for key, value in mapping.items():
    df[value] = df['Placenta'].apply(lambda x: 1 if key in x else 0)

    
# Define the pattern to match "AFI" followed by 10 characters
pattern = r'A.F(.{13})'

# Function to extract the matched strings
def extract_matched_strings(text):
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # Extract the 10 characters after "CER"
    else:
        return None

# Apply the function to the DataFrame column
df['Extracted_String_A.F.I'] = df['usg'].apply(extract_matched_strings)
df['Extracted_String_A.F.I']=df['Extracted_String_A.F.I'].astype(str)

df['A.F.I_upd']=df['A.F.I'].fillna(df['Extracted_String_A.F.I'])

# Regular expression to extract numerical values
pattern = r'(\d+\.?\d*)'

# Function to extract numerical values
def extract_numerical_values(text):
    matches = re.findall(pattern, str(text))
    return ' '.join(matches)

# Apply the function to the DataFrame column
df['A.F.I_numerical'] = df['A.F.I_upd'].apply(extract_numerical_values)

df['A.F.I_numerical']=df['A.F.I_numerical'].replace({'176 3 6.4':'6.4','14.0 4.2':'14.0','9 10':'9','10.3 3 2.8':'10.3',
                                                    '13 14':'13',
                                                    '2306 34':'ADEQUATE',
                                                    '112':'11.2','163':'16.3'})



# Define the pattern to match "CER" followed by 10 characters
pattern = r'CERVIX-(.{5})'

# Function to extract the matched strings
def extract_matched_strings(text):
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # Extract the 10 characters after "CER"
    else:
        return None

# Apply the function to the DataFrame column
df['Extracted_String_cervix'] = df['usg'].apply(extract_matched_strings)
df['Extracted_String_cervix']=df['Extracted_String_cervix'].astype(str)

df['cervix_upd']=df['Cervix_Length'].fillna(df['Extracted_String_cervix'])

# Regular expression to extract numerical values
pattern = r'(\d+\.?\d*)'

# Function to extract numerical values
def extract_numerical_values(text):
    matches = re.findall(pattern, str(text))
    return ' '.join(matches)

# # Apply the function to the DataFrame column
df['cervix_numerical'] = df['cervix_upd'].apply(extract_numerical_values)

df['cervix_numerical']=df['cervix_numerical'].replace({'2838 12':'normal','32 1719 251 12.3':'32',
                                                      '35':'3.5','41':'4.1','45':'4.5','36':'3.6','34':'3.4',
                                                      '33':'3.3','29':'2.9','32':'3.2','30':'3.0'})



## Code to update FHR


pattern = r'FHR-(\d+)'
import re
# Function to extract FHR value from a string
def extract_fhr(column_value):
    # Convert non-string values to string
    if not isinstance(column_value, str):
        column_value = str(column_value)
    fhr_match = re.search(pattern, column_value)
    if fhr_match:
        return fhr_match.group(1)
    else:
        return None

# Apply the function to the column
df['new_fhr']=df['FHR'].apply(extract_fhr)
df['fhr_upd'] = df['new_fhr'].fillna(df['FHR'])


df.drop(columns=[ 'nan','Extracted_String_A.F.I', 'A.F.I_upd','Extracted_String_cervix', 'cervix_upd','new_fhr'],inplace=True)

def keyword_extract(df,column,keyword):
    keywords_pattern = r'\b(' + keyword + r')\b'

    df['keywords'] = df[column].str.extract(keywords_pattern, expand=False)
    return df['keywords']


for i in ['OLIGOHYDRAMNIOUS',"OLIGO","FETAL DISTRESS","DISTRESS","FETAL","BREECH",'CEPHALIC','PREVIA',
         'ECLAMPSIA' , 'PPROM', 'PROM', 'PREVIA', 'PREECLAMPSIA','OBSTRICAL']:
    key  = i
    df["col_"+str(key)]=keyword_extract(df,"diagnosis",key)
    df["col_"+str(key)]=df["col_"+str(key)].fillna(keyword_extract(df,"indication",key))
    df["col_"+str(key)]=df["col_"+str(key)].fillna(keyword_extract(df,"usg",key))
    print( df["col_"+str(key)].value_counts())
    
    
first_keywords = [entry.split()[0] if entry else None for entry in df['diagnosis']]
extracted_keywords = [keyword.split("[")[1] if "[" in keyword else None for keyword in first_keywords]
cleaned_keywords = [keyword[1:] if keyword.startswith("'") else keyword for keyword in extracted_keywords]
cleaned_keywords = [keyword[:-1] if keyword.endswith("'") else keyword for keyword in cleaned_keywords]
df['diagnosis_1'] = cleaned_keywords
df['primi'] = np.where(df['diagnosis_1'].isin(['PRIMI','PRIMIGRAVIDA','PRI','PRIM','PTIMI','PRAMI']),1,0)
df['diagnosis_1']= df['diagnosis_1'].replace("nan',",np.NaN)
df['diagnosis_1']= df['diagnosis_1'].replace(" ",np.NaN)
df['check_dia'] = np.where((df['primi']==1) | (df['diagnosis_1']== 'ELDERLY'),np.NaN, df['diagnosis_1'])

def extract_alphabets(string):
    alphabet_counts = {}
    if pd.notna(string):
        for char in string:
            if char.isalpha():
                alphabet_counts[char] = 0
        for i in range(len(string) - 1):
            if string[i].isalpha() and string[i + 1].isdigit():
                alphabet_counts[string[i]] = int(string[i + 1])
    return alphabet_counts

# Extract unique alphabets and their counts
unique_alphabets = set()
for value in df['check_dia']:
    alphabet_counts = extract_alphabets(value)
    unique_alphabets.update(alphabet_counts.keys())

unique_alphabets
for alphabet in unique_alphabets:
    df[alphabet] = ''
# Fill in the columns with values following each alphabet
for i, value in enumerate(df['check_dia']):
    alphabet_counts = extract_alphabets(value)
    for alphabet, count in alphabet_counts.items():
        df.at[i, alphabet] = count
df[['A','G','L','D','P']] = df[['A','G','L','D','P']].replace('',0)
df = df.rename(columns={

    'G': 'gravida',
    'A' :'abortion',
    'L' :'living_children',
    'D' :'dead_children',
    'P' : 'parity'

})

df['col_FETAL DISTRESS']=df['col_FETAL DISTRESS'].fillna(df['col_FETAL'])
df['col_FETAL DISTRESS']=df['col_FETAL DISTRESS'].fillna(df['col_DISTRESS'])
df['col_OLIGOHYDRAMNIOUS']=df['col_OLIGOHYDRAMNIOUS'].fillna(df['col_OLIGO'])


col_drop=['col_DISTRESS', 'col_FETAL', 'diagnosis_1','check_dia','col_OLIGO']
df.drop(columns=col_drop,inplace=True)
