import numpy as np
import pandas as pd
import joblib


def preprocess_train(accounts,products,sales):
    products['product'] = products['product'].apply(lambda x: 'GTXPro' if x=='GTX Pro' else x)
    #Intergrating Datasets.
    df = pd.merge(accounts,sales,how='left',on=['account'])
    dataset = pd.merge(df,products,how='left',on=['product'])
    #filter data to work on won/lost deal stage
    dataset= dataset[dataset['deal_stage'].isin(['Won','Lost'])]
    data = dataset.drop(['opportunity_id','series','subsidiary_of'],axis=1)
    # #calc win_rate for sales agent to include it in data
    slicing_data = data[['sales_agent','deal_stage']]
    slicing_data['deal'] = slicing_data['deal_stage'].apply(lambda x: 1 if x=='Won' else 0)
    slicing_data = slicing_data.groupby('sales_agent').agg(
        won_deals=('deal', 'sum'),
        total_deals=('deal', 'count')
    )
    slicing_data['agent_win_rate'] = (slicing_data['won_deals'] / slicing_data['total_deals'])
    slicing_data.reset_index(inplace=True)
    agent_win_rate = {x:y for x,y in zip(slicing_data['sales_agent'],slicing_data['agent_win_rate'])}
    data['agent_win_rate'] = data['sales_agent'].apply(lambda x:agent_win_rate.get(x,-1))
    data.drop('sales_agent',axis=1,inplace=True)

    unique_accounts  = data['account'].unique()
    unique_sectors   = data['sector'].unique()
    unique_locations = data['office_location'].unique()
    unique_products  = data['product'].unique()
    unique_deal_stage  = data['deal_stage'].unique()



    account_dict  = {value: idx for idx, value in enumerate(unique_accounts,start=1)}
    sector_dict   = {value: idx for idx, value in enumerate(unique_sectors,start=1)}
    location_dict = {value: idx for idx, value in enumerate(unique_locations,start=1)}
    product_dict  = {value: idx for idx, value in enumerate(unique_products,start=1)}

    
    deal_stage_dict  = {value: idx for idx, value in enumerate(unique_deal_stage[::-1],start=0)}
   

    rev_account_dict  = {key:value for value, key in account_dict.items()}
    rev_sector_dict   = {key:value for value, key in sector_dict.items()}
    rev_location_dict = {key:value for value, key in location_dict.items()}
    rev_product_dict  = {key:value for value, key in product_dict.items()}
    rev_deal_stage_dict  = {key:value for value, key in deal_stage_dict.items()}

    #encoding account,office_location,sector,product
    data['account'] = data['account'].map(account_dict)
    data['office_location'] = data['office_location'].map(location_dict)
    data['sector'] = data['sector'].map(sector_dict)
    data['product'] = data['product'].map(product_dict)
    
    data['deal_stage'] = data['deal_stage'].map(deal_stage_dict)



    # Calculate duration
    #calc process_duration 
    data['engage_date'] = pd.to_datetime(data['engage_date'])
    data['close_date'] = pd.to_datetime(data['close_date'])
    data['process_duration'] = (data['close_date'] - data['engage_date']).dt.days


    #calc engage_month
    data['engage_month'] = data['engage_date'].dt.month

    #remove engage_date,close_date,close_value from data when analyzing to find out the root cause
    
    data_cleaned = data.drop(['close_date','close_value'],axis=1)

    data_check = data_cleaned.drop(['engage_date','account','year_established'],axis=1)
    df = data_check.copy()
    print('pickle preprocessing_encoding_dict...')
    joblib.dump([account_dict, sector_dict, product_dict, location_dict,deal_stage_dict,agent_win_rate], "preprocess_dicts.joblib")
    
    return df


def preprocess_test(accounts,products,sales):

    account_dict, sector_dict, product_dict, location_dict,_,agent_win_rate = joblib.load("preprocess_dicts.joblib")

    
    products['product'] = products['product'].apply(lambda x: 'GTXPro' if x=='GTX Pro' else x)

    #Intergrating Datasets.
    df = pd.merge(accounts,sales,how='left',on=['account'])
    dataset = pd.merge(df,products,how='left',on=['product'])
    #filter data to work on won/lost deal stage
    data = dataset.drop(['opportunity_id','series','subsidiary_of'],axis=1)
   
    data['agent_win_rate'] = data['sales_agent'].apply(lambda x:agent_win_rate.get(x,-1))
    data.drop('sales_agent',axis=1,inplace=True)

    data['account'] = data['account'].apply(lambda x:account_dict.get(x,-1))
    data['office_location'] = data['office_location'].apply(lambda x:location_dict.get(x,-1))
    data['sector'] = data['sector'].apply(lambda x:sector_dict.get(x,-1))
    data['product'] = data['product'].apply(lambda x:product_dict.get(x,-1))

    # rev_account_dict  = {key:value for value, key in account_dict.items()}
    # rev_sector_dict   = {key:value for value, key in sector_dict.items()}
    # rev_location_dict = {key:value for value, key in location_dict.items()}
    # rev_product_dict  = {key:value for value, key in product_dict.items()}


    data['engage_date'] = pd.to_datetime(data['engage_date'])

    #calc engage_month
    data['engage_month'] = data['engage_date'].dt.month

    #remove engage_date,close_date,close_value from data when analyzing to find out the root cause
    try:
        data = data.drop(['close_date','close_value'],axis=1)
    except:
        pass
    data_check = data.drop(['engage_date','account','year_established'],axis=1)

    return data_check














