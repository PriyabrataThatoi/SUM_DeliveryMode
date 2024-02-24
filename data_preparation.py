import numpy as np
import pandas as pd
import os
from functools import reduce
import pandas as pd
import re
import glob
from datetime import date

# Define the path to your 'results' folder
folder_path = os.getcwd()

# Create a pattern to match all .xlsx files
pattern = os.path.join(folder_path, '*.xlsx')

# Use glob.glob to find all files in the folder that match the pattern
xlsx_files = glob.glob(pattern)

# Now xlsx_files contains a list of full file paths. If you want just the filenames, you can do:
xlsx_filenames = [os.path.basename(file) for file in xlsx_files]

print(xlsx_filenames)



def extract_keyword(df,col,keyword) :
    # Reinitialize the list to store the data for weight extraction
    extracted_data = []

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        # Get the current 'SL.NO'
        current_sl_no = row['SL.NO']
        # If the 'DATE' column contains 'kg', extract the weight value
        if pd.notnull(row[col]) and keyword in str(row[col]):
            current_lmp = row[col].split(' ')[0] # Assuming the format is 'XX kg'
            # Append the SL.NO and weight to the list
            extracted_data.append({'SL.NO': current_sl_no, keyword: current_lmp})

    # Convert the list of dictionaries to a DataFrame
    keyword_df = pd.DataFrame(extracted_data)

    # Drop duplicate rows (if any) to have unique 'SL.NO' - 'weight' pairs
    # print(keyword_df)
    keyword_df = keyword_df.drop_duplicates()

    # Create a DataFrame with unique 'SL.NO' from the original DataFrame
    sl_no_df = df[['SL.NO']].drop_duplicates()

    # Merge this with the extracted weight data
    # Use an outer join to ensure all 'SL.NO' are included, even if there's no corresponding 'weight'

    complete_keyword_df = sl_no_df.merge(keyword_df, on='SL.NO', how='left')
    return complete_keyword_df

    # complete_lmp_df['lmp']= complete_lmp_df['lmp'].str.split('-').str[1]


def remove_nans(lst):
    return [x for x in lst if pd.notnull(x)]


def get_first_element(lst):
    if isinstance(lst, list) and lst:
        return lst[0]
    else:
        return None



def f_global_data_pref(filename,columns):
    '''
    function to prepare global dataframe with initial features
    :param filename: list of xlsx files
    :return: dataframe
    '''
    print(filename)
    df = pd.read_excel(filename)
    df.columns = columns
    print(df.shape)

    column_terminate = 19
    row_terminate = 2

    df_ = df.iloc[row_terminate:,:column_terminate]
    for col in df_.columns.tolist() :
        df_[col] = df_[col].astype(str)

    df_['SL.NO'] = pd.to_numeric(df_['SL.NO'], errors='coerce')
    df_['AGE'] = df_['AGE'].str.extract(r'(\d+)').astype(float)
    # Forward fill the 'SL.NO' column again
    df_['SL.NO'] = df_['SL.NO'].ffill()

    df_1= extract_keyword(df_,'REGISTRATION','hight-')
    df_1['hight-'] = df_1['hight-'].str.extract(r'(\d+)').astype(float)

    df_2= extract_keyword(df_,'DATE','kg')
    df_2['kg'] = df_2['kg'].str.extract(r'(\d+)').astype(float)

    df_3= extract_keyword(df_,'EDD/GA','LMP')
    df_3['LMP']=df_3['LMP'].str.split('-').str[1]

    df_4 = df_[['SL.NO','AGE']].drop_duplicates()
    df_4 = df_4[~df_4.AGE.isna()]

    print(df_4.head())
    # df_4= extract_keyword(df_,'EDD/GA','GA- ')

    df_5= extract_keyword(df_,'EDD/GA','EDD')
    df_5['EDD']=df_5['EDD'].str.split('-').str[1]

    # extract and combine diagnosis
    df_6 =  df_[~df_['SL.NO'].isnull()].groupby(['SL.NO']).agg({'INDICATION': list}).reset_index()
    df_6['INDICATION'] = df_6['INDICATION'].apply(lambda x : remove_nans(x))

    # extract and combine diagnosis
    df_7 =  df_[~df_['SL.NO'].isnull()].groupby(['SL.NO']).agg({'DIAGNOSIS': list}).reset_index()
    df_7['DIAGNOSIS'] = df_7['DIAGNOSIS'].apply(lambda x : remove_nans(x))

    # create sonography diagnosis
    df_8 = df_[~df_['SL.NO'].isnull()].groupby(['SL.NO']).agg({'                   USG': list}).reset_index()
    df_8['USG'] = df_8['                   USG'].apply(lambda x: remove_nans(x))
    df_8 = df_8.drop(columns='                   USG')

    # additional parameters
    df_9 = df_[['SL.NO', 'HB',
                'BP', '         HHH', '           B -G']].drop_duplicates().dropna(how='all', subset = ['HB',
                                                                                                        'BP', '         HHH', '           B -G'])
    df_9 = df_9[~(df_9.HB=='nan')]

    # GA Weeks
    extracted_gw_data = []
    for _, row in df_.iterrows():
        # Get the current 'SL.NO'
        current_sl_no = row['SL.NO']
        # If the 'DATE' column contains 'kg', extract the weight value
        if pd.notnull(row['EDD/GA']) and 'GA' in str(row['EDD/GA']):
            current_lmp = re.findall(r'\d+', row['EDD/GA'])  # Assuming the format is 'XX kg'
            # Append the SL.NO and weight to the list
            extracted_gw_data.append({'SL.NO': current_sl_no, 'GA': current_lmp})

    # Convert the list of dictionaries to a DataFrame
    gw_df = pd.DataFrame(extracted_gw_data)
    gw_df['GA'] = gw_df['GA'].apply(get_first_element)

    # Drop duplicate rows (if any) to have unique 'SL.NO' - 'weight' pairs
    df_10 = gw_df.drop_duplicates()

    a= df_[~df_['SL.NO'].isna()][['SL.NO','ELECTIVE/EMERGENCY']].drop_duplicates()
    a['rnk']= a.groupby(['SL.NO'])['ELECTIVE/EMERGENCY'].rank()
    df_11 =  a[a['rnk']==1].drop(columns='rnk')

    # create target variable
    # target_df = df_[~((df_['METHOD-VD/CS'].isnull()) | (df_['METHOD-VD/CS']=='nan'))][['SL.NO','METHOD-VD/CS']].drop_duplicates()
    # target_df['METHOD-VD/CS']= np.where((target_df['METHOD-VD/CS']!='LSCS'),'NVD',target_df['METHOD-VD/CS'])
    target_df = df_[~((df_['METHOD-VD/CS'].isnull()) | (df_['METHOD-VD/CS']=='nan'))][['SL.NO','METHOD-VD/CS']].drop_duplicates()
    target_df['METHOD-VD/CS']= np.where((target_df['METHOD-VD/CS']!='LSCS'),'NVD',target_df['METHOD-VD/CS'])
    target_df = target_df.dropna().drop_duplicates()


    # merge all dataframes:
    dataset=[df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,target_df]
    # Merge all DataFrames on 'SL.NO'

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='SL.NO', how='left'), dataset)

    # print(merged_df.isna().sum())
    merged_df = merged_df.rename(columns=
    {
        'SL.NO':'sl_no',
        'hight-':'height',
        'AGE':'age',
        'kg':'kg',
        'LMP':'lmp',
        'EDD':'edd',
        'INDICATION':'indication',
        'DIAGNOSIS':'diagnosis',
        'USG':'usg',
        'HB':'hb',
        'BP':'bp',
        '         HHH':'hhh',
        '           B -G':'bg',
        'GA':'ga_weeks',
        'METHOD-VD/CS':'delivery_mode',
        'ELECTIVE/EMERGENCY':'elective_emergency'
    })
    merged_df['filename'] = filename
    print(merged_df.shape, merged_df['sl_no'].nunique())


    print(f'data processed for :{filename}, final output : {merged_df.shape} ')
    print()
    print('---------------------')
    return merged_df


# columns = df.columns
columns=[                             'SL.NO',
                                      'DATE',
                                      'REGISTRATION',
                                      '    NAME  &  ADDRESS',
                                      'AGE',
                                      'SES/BPL/APL',
                                      'EDUCATION',
                                      'OCCUPATION',
                                      'REFRAL',
                                      'EDD/GA',
                                      'DIAGNOSIS',
                                      'METHOD-VD/CS',
                                      'ELECTIVE/EMERGENCY',
                                      'INDICATION',
                                      'HB',
                                      'BP',
                                      '         HHH',
                                      '           B -G',
                                      '                   USG',
                                      'DRS NAME & UNIT',
                                      'BABY--TOB',
                                      'DOB',
                                      'SEX',
                                      'WT',
                                      'APGAR',
                                      'NICU',
                                      'INDICATION',
                                      'TRANS-OUT-BABY',
                                      'INDUCTION',
                                      ' DT  OF   DISCHARGE',
                                      'MOTHER CONDITION',
                                      'B/O   IP NO',
                                      'GDM',
                                      'THYROIED',
                                      'RH-VE',
                                      'OTHER',
                                      'BLOOD TRANSFUSE/ RECIVED, RFF, RDP',
                                      'CLOT/UN CLOT'
                                      ]

print(len(columns))
dataframes = []

for filename in xlsx_filenames:
    temp = f_global_data_pref(filename,columns)
    dataframes.append(temp)

df_delivery = pd.concat(dataframes,ignore_index=True)
# df_delivery = df_delivery.drop_duplicates()
df_delivery = df_delivery.reset_index().rename(columns={'index':'patient_id'}).drop(columns='sl_no')
df_delivery = df_delivery[df_delivery.patient_id>0]
print(df_delivery.shape)
# df_delivery['delivery_mode'][df_delivery.delivery_mode=='VACCUME'] = 'NVD'
# df_delivery = df_delivery.drop_duplicates()


str_date = str(date.today())
print('output dataset shape: ',df_delivery.shape)
print(folder_path+'/res/'+'df_delivery_mode_'+str_date+'.csv')
df_delivery.to_csv(folder_path+'/res'+'df_delivery_mode_'+str_date+'.csv',index=False)