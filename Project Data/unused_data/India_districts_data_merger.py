
import pandas as pd

df = pd.read_csv(r"FinalDataIndia.csv")
df1 = pd.read_csv(r"India_districts_population_data.csv")

print(df)

print(df1)


df2_merged = pd.merge(df,df1,left_on='District', right_on='District')


df2_cleaned = df2_merged[['Date','State_x','District','Confirmed','Recovered','Deceased','Tested','Population']]

print(df2_cleaned.columns)

df2_cleaned.to_csv(r"India_data_cleaned.csv")