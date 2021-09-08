from wrangler.utils import get_attributes
import pandas as pd
from datetime import datetime, date
from IPython.display import display
import numpy as np
import yaml
import re,os
from collections import Counter
from numpy import percentile
import sys
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def age(data):
    
    """
    Convert the birthdate of individuals to their age presently
    Parameters
    -----------
    data: DataFrame or name Series.
        Dataframe column to perform operation on.
    Returns
    -------
    New dataframe with the age of the user (int)
    """
    df = data.copy()
    df['customer_age'] = df['checkout_date_checkout'] - pd.to_datetime(df['customer_birth_date'])
    df['customer_age'] = df['customer_age'].fillna(df['customer_age'].mean())
    df['customer_age'] = ((df['customer_age'].astype(str).str.split(' ').str[0].astype(int))/365).round()
    return df

def bin_age(dataframe=None, age_col=None, add_prefix=True):

    """
    The age attribute in a DataFrame is binned into 5 categories:
    (baby/toddler, child, young adult, mid-age and elderly).
    Parameters
    -----------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    age_col: the name of the age column in the dataset. A string is expected
        The column to perform the operation on.
        
    add_prefix: Bool. Default is set to True
        add prefix to the column name. 
    Returns
    -------
    Dataframe with binned age attribute
    """
    if dataframe is None:
        raise ValueError("dataframe: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(age_col, str):
        errstr = f'The given type for age_col is {type(age_col).__name__}. Expected type is a string'
        raise TypeError(errstr)
        
    data = dataframe.copy()
    
    if add_prefix:
        prefix_name = f'binned_{age_col}'
    else:
        prefix_name = age_col
    
    bin_labels = ['Toddler/Baby', 'Child', 'Young Adult', 'Mid-Age', 'Elderly']
    data[prefix_name] = pd.cut(data[age_col], bins = [0,2,17,30,45,99], labels = bin_labels)
    data[prefix_name] = data[prefix_name].astype(str)
    return data

def check_nan(dataframe=None, plot=False, verbose=True):
    
    """
    Display missing values as a pandas dataframe and give a proportion
    in terms of percentages.
    Parameters
    ----------
    data: DataFrame or named Series
    
    plot: bool, Default False
        Plots missing values in dataset as a heatmap
        
    verbose: bool, Default False
            
    Returns
    -------
    Matplotlib Figure:
        Heatmap plot of missing values
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    data = dataframe.copy()
    df = data.isna().sum()
    df = df.reset_index()
    df.columns = ['features', 'missing_counts']

    missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 1)
    df['missing_percent'] = missing_percent
    nan_values = df.set_index('features')['missing_percent']
    
    print_divider('Count and Percentage of missing value')
    if plot:
        plot_nan(nan_values)
    if verbose:
        display(df)
    check_nan.df = df

def clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Clean relevant columns in the dataset to complete pre-processing
    Parameters
    -----------
    data: DataFrame or name Series.
        Dataframe column to perform operation on.
    Returns
    -------
    New dataframe with the cleaned columns
    '''
    
    df = data.copy()
    
    #converting published date column to datetime format
    df['book_publishedDate'] = pd.to_datetime(df['book_publishedDate'])

    #cleaning other numerical columns
    columns_to_clean = ['book_price', 'book_pages', 'customer_zipcode', 'library_postal_code']
    for column in df[columns_to_clean]:
        df[column] = df[column].replace("[^0-9]", '', regex = True)
        df[column] = df[column].astype('float').astype('Int64')
        df[column] = df[column].fillna(0).astype(np.int64, errors='ignore')

    #creating age of book (how old the book is since its publishing)
    df['book_age'] = df['checkout_date_checkout'] - df['book_publishedDate']
    df['book_age'] = df['book_age'].fillna(df['book_age'].mean())
    df['book_age'] = ((df['book_age'].astype(str).str.split(' ').str[0].astype(int))/365).round()
    
    #Round up customer age and book age to the nearest integar
    df['customer_age'] = df['customer_age'].round(0)
    df['book_age'] = df['book_age'].round(0)
    
    #more cleaning
    for column in df[['book_authors', 'book_categories']]:
        df[column] = df[column].str.strip("[']")
        
    #creating an all text column before generating text embeddings
    df['all_text'] = df.book_title.fillna('').astype(str)+ " " + \
    df.book_authors.fillna('').astype(str) + " " + df.book_categories.fillna('').astype(str)
    
    #Extracting specific names from the Multnomah Libraries
    df['library_name'] = df['library_name'].str.lower()
    df['library_location'] = df['library_name'].str.split('library').str[-1]
    df['library_location'] = df['library_location'].replace('', 'unknown')
    df['library_location'] = df['library_location'].str.strip()
    
    #Cleaning the customers' education qualification
    cols_to_clean = ['customer_education', 'customer_state', \
                                  'customer_gender', 'library_city','library_region', \
                                  'customer_occupation', 'customer_city']
    df = remove_excess_space(df, cols_to_clean)
    for col in cols_to_clean:
        df[col][df[col] == 'nan'] = df[col].value_counts().index[0]

    return df

def create_target_column(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Create the target column for the Machine Learning model to be built
    Parameters
    -----------
    data: DataFrame or name Series.
        Dataframe column to perform operation on.
    Returns
    -------
    New dataframe with the target column (int)
    '''
    df = data.copy()
    
    df['period'] = df['checkout_date_returned'] - df['checkout_date_checkout']
    df['period'] = df['period'].astype(str).str.split(' ').str[0]
    df['period'] = df['period'].astype(int)
    df.drop(df[df['period'] < 0].index, inplace = True)
    target = []
    for row in df['period']:
        if row <= 28:
            target.append(0)
        else:
            target.append(1)
    df['target'] = target
    
    return df
    
def detect_fix_outliers(dataframe=None,target_column=None,n=1,num_features=None,fix_method='mean',verbose=True):
        
    '''
    Detect outliers present in the numerical features and fix the outliers present.
    Parameters:
    ------------------------
    data: DataFrame or name Series.
        Data set to perform operation on.
    num_features: List, Series, Array.
        Numerical features to perform operation on. If not provided, we automatically infer from the dataset.
    target_column: string
        The target attribute name. Not required for fixing, so it needs to be excluded.
    fix_method: mean or log_transformation.
        One of the two methods that you deem fit to fix the outlier values present in the dataset.
    Returns:
    -------
        Dataframe
            A new dataframe after removing outliers.
    
    '''

    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'") 

    data = dataframe.copy()
    
    df = data.copy()
    
    outlier_indices = []
    
    if num_features is None:
        if not isinstance(target_column,str):
            errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
            raise TypeError(errstr) 
        num_attributes, cat_attributes = get_attributes(data,[target_column])
    else:
        num_attributes = num_features

    for column in num_attributes:
        
        data.loc[:,column] = abs(data[column])
        mean = data[column].mean()

        #calculate the interquartlie range
        q25, q75 = np.percentile(data[column].dropna(), 25), np.percentile(data[column].dropna(), 75)
        iqr = q75 - q25

        #calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower,upper = q25 - cut_off, q75 + cut_off

        #identify outliers
        # Determine a list of indices of outliers for feature col
        outlier_list_col = data[(data[column] < lower) | (data[column] > upper)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
        
        #apply any of the fix methods below to handle the outlier values
        if fix_method == 'mean':
            df.loc[:,column] = df[column].apply(lambda x : mean 
                                                        if x < lower or x > upper else x)
        elif fix_method == 'log_transformation':
            df.loc[:,column] = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        else:
            raise ValueError("fix: must specify a fix method, one of [mean or log_transformation]")

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
    if verbose:
        print_divider('Table identifying Outliers present')
        display(data.loc[multiple_outliers])

    return df
    
def drop_cols(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    
    '''
    Drop redundant columns in the dataframe
    '''
    data = data.drop(columns,axis=1)
    return data
    
def featurize_datetime(dataframe: pd.DataFrame, columns: list, drop: bool =True) -> pd.DataFrame:
    '''
    Featurize datetime in the dataset to create new fields 
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
    column: The columns to perform the operation on. A string is expected
    suffix: The name to append to the column
    drop: Bool. Default is set to True
        drop original datetime column. 
    Returns
    -------
        Dataframe with new datetime fields
            
    '''
    
    df = dataframe.copy()
    
    for column in columns:
        pattern = '|'.join(["[^0-9]", '%'])
        df[column] = df[column].str.replace(pattern, '', regex=True)
        df[column] = df[column].str.replace("/", '-')
        fld = df[column]

        if not np.issubdtype(fld.dtype, np.datetime64):
            df.loc[:,column] = fld = pd.to_datetime(fld, infer_datetime_format=True,utc=True).dt.tz_localize(None)

        df.loc[:,'{}_year'.format(column)] = df.loc[:,column].dt.year
        df.loc[:,'{}_month'.format(column)] = df.loc[:,column].dt.strftime('%B')
        df.loc[:,'{}_day'.format(column)] = df.loc[:,column].dt.day_name()
    
    
    if drop: df.drop([column], axis=1, inplace=True)
        
    return df

def handle_nan(data,strategy='mean',fillna='mode'):
    
        
    """

    Function handles NaN values in a dataset for both categorical
    and numerical variables

    Args:
        data: DataFrame to perform operation on
        strategy: Method of filling numerical features
        fillna: Method of filling categorical features
    """
    df = data.copy()
    
    num_attributes, cat_attributes = get_attributes(data,['target'])
    if strategy=='mean':
        for item in df[num_attributes]:
            df[item] = df[item].fillna(df[item].mean())
    if fillna == 'mode':
        for item in df[cat_attributes]:
            df[item] = df[item].fillna(df[item].value_counts().index[0])
    else:
        for item in df[num_attributes]:
            df[item] = df[item].fillna(fillna)

    return df

    
     
def main(data):
    
    '''
    Main function that handles the end to end of data cleaning and featurization
    
    Parameters:
    ------------------------
    data: DataFrame or name Series.
        Dataset to perform operation on.
    
    Returns
    -------
        Dataframe with new featurized and cleaned columns
    '''
    
    df = data.copy()
    cols_to_drop = ['book_id','book_title','book_authors',\
                'book_publishedDate', 'book_categories','checkout_date_checkout',\
               'checkout_date_returned', 'customer_nam', 'customer_street_address',\
               'customer_zipcode', 'customer_birth_date', 'library_id',\
               'library_name', 'library_street_address', 'library_postal_code',\
                'checkout_id','checkout_patron_id','checkout_library_id', 'book_publisher']
    
    
    df = featurize_datetime(df, ['checkout_date_checkout', 'checkout_date_returned'], drop = False)
    df = age(df)
    df = create_target_column(df)
    df = clean_columns(df)
    df = detect_fix_outliers(df, target_column='target', verbose=False)
    df = handle_nan(df)
    df = bin_age(df, age_col = 'customer_age')
    df = drop_cols(df, columns=cols_to_drop)
        
    return df

def print_divider(title: str):
    '''
    Print formatting to make print statements look cool.
    '''
    print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))
    
def remove_excess_space(data: pd.DataFrame, columns: list):
    
    '''
    Remove excess space from specified columns
    
    Parameters:
    ------------------------
    data: DataFrame or name Series.
        Dataset to perform operation on.
    columns: The list of columns to remove excess space from. A list is expected
    
    Returns
    -------
        Dataframe with new column names
    '''
    df = data.copy()
    for column in columns:
        df[column] = df[column].str.lower()
        cleaned_text = []
        for row in df[column]:
            step_1 = re.sub(' +', ' ', str(row))
            step_2 = step_1.strip()
            cleaned_text.append(step_2)
        df[column] = cleaned_text
    return df

def rename_columns(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    
    '''
    Rename overlapping column names to create unique column names  
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Dataset to perform operation on.
    prefix: The prefix to add to column names. A string is expected
    
    Returns
    -------
        Dataframe with new column names
    '''
    data = data.add_prefix(prefix)
    return data

if __name__ == '__main__':
    main(data)