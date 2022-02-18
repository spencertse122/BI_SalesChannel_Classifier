# Importing 
from models import data_processing
import pandas as pd
from pandas.api.types import is_string_dtype

# Picking a random file to support testing process
test_data = "../RawData/Analytics_Data1.txt"
test_data_foler = "../RawData"

# loading in the data
houseOfData= data_processing.house_of_data(test_data_foler)


def test_read_file():
    """
    Check if the data frame got loaded correctly
    """
    df = data_processing.read_file(test_data)

    # Check if it's a dataframe
    assert type(df) == pd.core.frame.DataFrame
    # Check if the len of data frame matches
    assert len(df) == 860000

def test_read_all_files():
    """
    Check if the function to load in everything within a directory works.
    """
    df = data_processing.read_all_files(test_data_foler)
    # Check if it's a dataframe
    assert type(df) == pd.core.frame.DataFrame
    # Check if the len of data frame matches
    assert len(df) == 1734110

def test_house_of_data_txt():
    """
    Testing the scenario to read txt.
    """
    houseOfData= data_processing.house_of_data(test_data)
    assert houseOfData._DataInput == test_data
    # Check if it's a dataframe
    assert type(houseOfData._RawData) == pd.core.frame.DataFrame
    # Check if the len of data frame matches
    assert len(houseOfData._RawData) == 860000

def test_house_of_data_folder():
    """
    Testing the scenario to read txt.
    """
    houseOfData= data_processing.house_of_data(test_data_foler)
    assert houseOfData._DataInput == test_data_foler
    # Check if it's a dataframe
    assert type(houseOfData._RawData) == pd.core.frame.DataFrame
    # Check if the len of data frame matches
    assert len(houseOfData._RawData) == 1734110

def test_house_of_data_cleaning_duplicates():
    """
    Checking if there's any duplicates
    """
    # loading in the data
    houseOfData= data_processing.house_of_data(test_data_foler)
    # Perfoming the cleaning
    houseOfData.cleaning()
    assert len(houseOfData._clean_df) == 1721310


def test_house_of_data_cleaning_whitespace():
    """
    Checking if there's any duplicates
    """
    # Perfoming the cleaning
    houseOfData.cleaning()

    # Creating a copy of the dataframe to perform the same cleaning again
    temp_df = houseOfData._clean_df.copy()
    for i in temp_df.columns:
        if is_string_dtype(temp_df[i]):
            # perform cleaning and compare the before after
            checker = temp_df[i].str.strip().values != temp_df[i].values  
            check = temp_df[checker]
            # Assert the before after are the same
            assert len(check) == 0
        else:
            pass

def test_house_of_data_cleaning_city_state():
    """
    validating if the cleaning works for below.
    e.g. 'KNOXVILLE, TN, TN' -> 'KNOXVILLE, TN'
    e.g. '95678, CA' -> 'ROSEVILLE, CA'
    """
    # Perfoming the cleaning
    houseOfData.cleaning()
    # Cleaning the KNOXVILLE, TN into one entity
    mask1 = houseOfData._clean_df['city_state'].str.contains("KNOXVILLE, TN, TN")
    assert len(houseOfData._clean_df[mask1]) == 0
    SplitCounts = len(houseOfData._clean_df['city_state'].str.split(", ", expand=True).columns)
    assert SplitCounts == 2

    # Checking the zip codes
    mask2 = houseOfData._clean_df['city_state'].str.split(", ", expand=True)[0].str.isnumeric()
    assert len(houseOfData._clean_df.loc[mask2, "city_state"]) == 0

# def test_house_of_data_other_data_validations_True():
#     """
#     See if the validation works
#     """
#     houseOfData.cleaning()
#     checks = houseOfData.other_data_validations()
#     assert type(checks) == list
    # assert False not in checks 

# def test_house_of_data_other_data_validations_False():
#     """
#     See if the validation works
#     """
#     houseOfData.cleaning()
#     checks = houseOfData.other_data_validations()
#     assert False in checks 

def test_house_of_data_enrichment():
    """
    Check to see enrichment has any errors.
    """
    houseOfData.enrichment()
    df = houseOfData._enriched_df

    # Initiate a filter to show inefficient sale that we can neglect.
    mask = df['firm_x_sales'] > 0
    mask1 = df['firm_x_sales'] < 2161.968 # this is the amount found in exploration as 20 precentile
    assert df.loc[mask & mask1, 'effective_sale'].unique() == 0

    # Validating if the driver pattern reflects the actual columns 
    mask1 = df['email_open'] == 0
    mask2 = df['web_visit'] == 0
    mask3 = df['webcast_attendee'] == 0
    mask4 = df['marketing_engaged'] == 0

    assert df[mask1 & mask2 & mask3 & mask4].equals(df[df['driver_pattern'] == (0,0,0,0)]) == True