import pandas as pd

try:
    data= pd.read_csv("LondonSmartMeter_Data_filtered2013nonull.tsv", sep= "\t")
except:
    pass

print (data.info())
acorn_df= data.iloc[:3].T
acorn_df.columns= acorn_df.iloc[0]
acorn_df = acorn_df.iloc[2:]

print(acorn_df.head(10))
print(acorn_df.head(10))

acorn_df.to_csv("acorn_df.csv")
df = data.iloc[4:]
df= df.rename(index=str, columns={"HouseholdID": "time"})

print(df.head())

df.to_csv("London_full_data.csv")

