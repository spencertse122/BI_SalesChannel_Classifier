"""
This is a module that houses all the pandas data manipulation process, including cleaning, encoding, 
shape changing, and more. The purpose of separating the following functionalities, is to ensure the code 
is reusable, and to support scalable future projects. Each sub-class is a particular view for a specific 
usage. (e.g.  feed_data_decision_tree is a machine learning model ready setup)
"""

# Importing libraries
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np

import os
import glob
from datetime import datetime 



cleaningDict = {
    '64150, MO' : 'RIVERSIDE, MO',
    '95678, CA' : 'ROSEVILLE, CA',
    '15801, PA' : 'DUBOIS, PA'
}

"""
Initiating all the reference data first for usages. Creating a reference data for cleaning zip code manually for now
since we don't have an approved API for this project.
"""

binary_encoding_map = {
    'Y' : 1,
    'N' : 0,
    'Engaged' : 1,
    'Not engaged' : 0
}

"""
Intiating the mapping for conversion
"""

driver_columns = [
    'email_open', 'web_visit', 
    'webcast_attendee', 'marketing_engaged'
    ]

"""
Creating a binary column list for ease of usages
"""

#Creating all the data loading methods
# Defining data loading functions

def read_file(data_file):
    """
    Function to read one data file with the custom setup
    """
    df = pd.read_csv(data_file, delimiter=";")
    return df

def read_all_files(data_folder):
    """
    Function to get all the data files of a target type, and
    """
    # Scanning for all the data files
    dataPaths = glob.glob(data_folder + "/*.txt")

    # Loading in the data
    listOfFrames = []
    for i in dataPaths:
        tdf = pd.read_csv(i, sep=";")
        tdf['source_file'] = i  # Adding source file as a column for ease of tracking
        listOfFrames.append(tdf)
        
    # Combining all the dataframes
    df = pd.concat(listOfFrames, ignore_index=True)
    return df


# Using a dictionary to map input methods
input_methods = {
    'txt' : read_file, # if it's a txt file, just use read csv
    '' : read_all_files # if it's a directory, use glob to read everything
}


class house_of_data(object):
    """
    Collection of data with each method as a stage of data manipulation process or metrics output.
    """

    def __init__(self, DataInput):
        # first validating the input to see which method to use
        inputType = os.path.splitext(DataInput)[-1]
        if inputType == ".txt":
            read_method = input_methods.get('txt')
        else:
            read_method = input_methods.get('')

        # Actually initiating the object attributes
        self._DataInput = DataInput
        self._RawData = read_method(DataInput)

    def cleaning(self):
        """
        Performing all the cleanings related to this dataset
        """
        df = self._RawData.drop_duplicates().copy() #Creating a copy of the drop_dup dataframe
        # Run a for-loop to go through string columns
        # and strip the leading and trailing whitespaces

        for i in df.columns:
            if is_string_dtype(df[i]):   # if it's a string column
                df[i] = df[i].str.strip()   # strip out the white spaces
            else:
                pass
    
        # resetting index after the drop duplicates
        df = df.reset_index(drop=True)

        # Cleaning up the duplicated states in cell
        df['city_state'] = df['city_state'].str.replace("WASHINGTON, DC, DC", "WASHINGTON DC, DC")
        df['city_state'] = df['city_state'].str.replace("KNOXVILLE, TN, TN", "KNOXVILLE, TN")

        # Performing the zip code cleaning
        df['city_state'] = df['city_state'].replace(cleaningDict)

        # Saving it to the object
        self._clean_df = df

    def other_data_validations(self):
        """
        Ensure new incoming data does not have new violations or needed updates.
        """
        checker = []
        # Checking if the broker ID is number
        checker.append(is_numeric_dtype(self._clean_df['broker_name'].str.split("Broker", expand=True)[1].astype(float)))
        # Checking if the broker name is correct
        checker.append(len(self._clean_df['broker_name'].str.split("Broker", expand=True).columns) == 2)
        # Checking if the city states only has one comma
        checker.append(len(self._clean_df['city_state'].str.split(", ", expand=True).columns) == 2)
        # Checking if the city state column contains numbers (zip codes)
        mask = self._clean_df['city_state'].str.split(", ", expand=True)[0].str.isnumeric()
        checker.append(len(self._clean_df[mask]) == 0)
        # Checking if thet prefix is a character per describe
        mask1 = self._clean_df['territory'].str[0].str.isalpha() == False
        mask2 = self._clean_df['territory'].str[0] != "I"
        mask3 = self._clean_df['territory'].str[0] != "W"
        checker.append(len(self._clean_df.loc[mask1, 'territory']) == 0)
        checker.append(len(self._clean_df.loc[mask2&mask3, 'territory']) == 0)
        # Checking fund category column
        mask4 = self._clean_df['fund_category'].str.isnumeric() == True
        checker.append(len(self._clean_df.loc[mask, 'fund_category']) == 0)
        checker.append(is_numeric_dtype(self._clean_df['firm_x_sales']))
        checker.append(is_numeric_dtype(self._clean_df['total_industry_sales']))

        # Checking binary columns
        bin_check = True
        for i in driver_columns:
            if len(self._clean_df[i].unique()) == 2:
                pass
            else:
                bin_check = False
                print(f"{i} is not binary")
                break

        checker.append(bin_check)

        return checker


    def enrichment(self):
        """
        Built on top of the clean_df, this is going to split out the analytics
        columns
        """
        # Creating a copy of the clean dataframe for enrichment
        self.cleaning()
        df = self._clean_df.copy()

        # Binary encode all the binary categorical variables
        df[driver_columns] = df[driver_columns].replace(binary_encoding_map)
        
        # Splitting out the sates since the city_state information is too granular for overview
        df['state'] = df['city_state'].str.split(", ", expand=True)[1]
        
        # Splitting out the Channel
        df['i_or_w'] =df['territory'].str[0]
        I_OR_W = {
            "I" : 1,
            "W" : 0
        }

        df['i_or_w'] = df['i_or_w'].replace(I_OR_W)
        
        # Initiating a filter to exclude all the no sales generated record
        mask = df['firm_x_sales'] > 0
        significant_cut = df[mask]['firm_x_sales'].quantile(0.2)
        # Intiating a filter with the twenty percentile cut
        mask1 = df['firm_x_sales'] > significant_cut
        # Performing the filter and label
        df['effective_sale'] = 0
        df.loc[mask1,'effective_sale'] = 1
        
        # Since driver_columns are the same as driver columns
        # we will utilize that to create driver_pattern
        df['driver_pattern'] = df[driver_columns].apply(tuple,axis=1)
        
        df = df.reset_index(drop=True)
        # Save the enriched dataframe to the house
        self._enriched_df = df
    



    def get_metrics(self):
        """
        Printing an output of quick metrics
        """
        pass




if __name__ == "__main__":
    import sys
    script, data_input = sys.args