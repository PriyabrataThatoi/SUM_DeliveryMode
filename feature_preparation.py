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
df['category']=df['Placenta']


# Remove leading and trailing spaces
df['category'] = df['category'].str.strip()

# Remove punctuation marks
df['category'] = df['category'].str.replace(r'[^\w\s]', '')

# Convert text to lowercase
df['category'] = df['category'].str.lower()

# Map similar categories to standardized categories
standard_categories = {
    'fundal': ['fundal', 'fundal anterior', 'fundal posterior', 'fundic', 'fundant in location',
               'fundal left anterior', 'fundal posterior with left lateral', 'fundal right lateral'],
    'anterior': ['anterior', 'fundoanterior cervix32 cm', 'anterior fundal', 'anterior fundic',
                 'anterior to the right not low lying', 'anterior and left lateral', 'anterior right fundal',
                 'anterior left lateral'],
    'posterior': ['posterior', 'posterior not low lying', 'posterior fundal afi113 cm', 'posterior rtlateral',
                  'posterior low lying', 'posterior left lateral', 'posteriorfundal', 'posterior fundal afi 10 cm',
                  'posterior and fundal', 'posterolaterally'],
    'left lateral': ['left lateral', 'left fundal', 'left anterior', 'left', 'left lateral posterior'],
    'right lateral': ['right lateral', 'right anterior', 'right']
}

# Function to map categories
def map_categories(category):
    for standard_category, variants in standard_categories.items():
        if category in variants:
            return standard_category
    return category

# Apply mapping function
df['category'] = df['category'].apply(map_categories)




# Map similar categories to standardized categories
category_mapping = {
    'fundal': ['fundal'],
    'fundoanterior': ['fundoanterior', 'fundoposteriorwithleftlateral', 'fundoanteriorcervix32cm', 
                      'fundoanteriorrightlateralextensionnotlowlying', 'fundoanteriornotlowlying', 
                      'fundanterior'],
    'posterior': ['posterior', 'posteriorfundalafi113cm', 'posteriorusginlr', 'posteriorfundal', 
                  'notlowlyinyposterior', 'rightlateralposterior', 'laterlalposterior', 'fundpposterior'],
    'anterior': ['anterior', 'anteriorfundal', 'anteriorfundic', 'anteriorafi10cm', 'anteriortotherightnotlowlying', 
                 'singlerightlateral', 'anteriorleftlateral'],
    'leftlateral': ['leftlateral'],
    'rightlateral': ['rightlateral', 'fundicrightlateral'],
    'lateral': ['lateral'],
    'separation': ['separation'],
    'nan': ['nan', ''],
    'other': []  # Placeholder for any unmatched categories
}

# Function to map categories
def map_categories(category):
    for standard_category, variants in category_mapping.items():
        if category in variants:
            return standard_category
    return 'other'

# Apply mapping function
df['category_placenta'] = df['category'].apply(map_categories)

#%%
df.head()
