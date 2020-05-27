"""
Script : data_process.py
Definition: This script is written for the following functionality :
    1. DB connection.
    2. Data Fetching from DB
    3. Dummy data creation if needed
    4. Creation of new features like dauly consumption, weekly consumption etc.
    5. Data pre processing.

"""

# import libraries
import pandas as pd  # version 0.23.4 , because of some bugs in the latest version 0.24
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import copy
from sklearn import preprocessing
from math import sqrt



"""
Function : create_new_features
Definition: this function takes the df1 argument, and does the following:
           1. creates DataFrame of daily_consumption, weekly_consumption, and monthly_consumption from df1.

Return: DataFrame of transposed daily_consumption, weekly_consumption, and monthly_consumption

"""


def create_new_features(df):
    # print(df["time"].info())

    # df = pd.read_csv("dataset_with_new_features.csv")

    df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%d')

    df = df.set_index("time")

    df_daily_consumption = pd.DataFrame()
    df_weekly_consumption = pd.DataFrame()
    df_monthly_consumption = pd.DataFrame()

    for i in df:
        df_daily_consumption["daily_total_consumption_" + str(i)] = df[i].resample("D").sum()
        df_weekly_consumption["weekly_total_consumption_" + str(i)] = df[i].resample("W").sum()
        df_monthly_consumption["Monthly_total_consumption_" + str(i)] = df[i].resample("M").sum()
    # df2['day'] = df["time"].dt.day_name()

    df_daily_consumption.T.to_csv("featureSet_daily.csv")
    df_weekly_consumption.to_csv("featureSet_weekly.csv")
    df_monthly_consumption.to_csv("featureSet_monthly.csv")

    return df_daily_consumption.T, df_weekly_consumption, df_monthly_consumption


"""
Function : create_main_dataFrame
Definition: this function takes the df1, daily consumption, weekly consumption and monthly consumption as argument, and 
does the following:
           1. create a main Dataframe where every rows are indexed bt the time series names.
           2. Create new columns contain 
                 a. daily consumptions for each day (May not be used)
                 b. diff between two consecutive day's consumptions(consumption b/w Monday and Tuesday) (May not be used)
                 c. Weekly consumptions 
                 d. diff between two consecutive week's consumption
                 e. Monthly consumptions 
                 f. diff between two consecutive month's consumption
Return: Main_df

"""


def create_main_dataFrame(df1, df_daily, df_weekly, df_monthly):
    Main_df = pd.DataFrame()
    #print(df1.info())
    # df["DayofWeek"] = df["time"].dt.dayofweek

    # df1 = df1.reset_index()
    df1 = df1.drop(["time"], axis=1)

    #print(df1.head(5))

    Main_df["Household_Name"] = df1.columns
    for idx in df_weekly.index:
        Main_df["Weekly_Consumption_" + str(idx)] = list(df_weekly.loc[idx])

    for idx in df_monthly.index:
        Main_df["Monthly_Consumption_" + str(idx)] = list(df_monthly.loc[idx])

    for i in range(1, len(Main_df.columns)):
        for j in range(i + 1, len(Main_df.columns)):
            Main_df["diff_bet_" + str(Main_df.columns[i] + "_nd_" + str(Main_df.columns[j]))] = Main_df[Main_df.columns[
                j]] - Main_df[Main_df.columns[i]]
            break
    """
    print(df_daily.info())
    df_daily= df_daily.reset_index()
    df = pd.DataFrame()
    for i in range(1, len(df_daily.columns)):
        for j in range(i + 1, len(df_daily.columns)):
            print(df_daily[df_daily.columns[j]] - df_daily[df_daily.columns[i]])
            Main_df["diff_bet_daily_consump_of_" + str(df_daily.columns[i]) + "_nd_" + str(df_daily.columns[j])] = df_daily[df_daily.columns[j]] - df_daily[df_daily.columns[i]]
            break
    """
    Main_df.to_csv("Main_df.csv")
    return Main_df


"""
Function :  data_preprocessing
Definition: this function takes 'df1' as argument, and does the following:
            1. encodes the column with categorical data/ object , i.e. date column 
            2. fill the missing values with the mean values.
            3. Save the database after the above tasks , named 'influxData.csv' 
Return: processed dataset named 'df1'

"""


def data_preprocessing(df1,data_choice):
    df1 = df1.set_index("Household_Name")

    # Encoding Object dtypes
    df1 = pd.get_dummies(df1, prefix_sep='_', drop_first=True)
    # le = preprocessing.LabelEncoder()

    # le.fit_transform(df1[])

    # missing values imputation
    # df1.dropna(inplace=True)
    df1.fillna(df1.mean(), inplace=True)

    cols = df1.columns

    #print(cols)
    scaler = MinMaxScaler()
    df1 = scaler.fit_transform(df1)
    df1 = pd.DataFrame(df1, columns=cols)

    #print(df1.shape)
    # df1["mean_Int_value"]= df

    #print(df1.info())
    # df1.drop(df1.select_dtypes(["datetime64[ns, UTC]"]), inplace=True, axis=1)

    df1.to_csv("dataset_preprocessed_"+str(data_choice)+".csv")

    return df1

