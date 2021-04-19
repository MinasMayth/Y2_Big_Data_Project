# Load pandas
import pandas as pd

# Read CSV file into DataFrame df
df = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Project Data\OxCGRT_latest.csv')

# Show dataframe
# print(df)



current_clustering_data = df.loc[(df.Date == 20210211)]

print(current_clustering_data)

current_clustering_data.to_excel("output1.xlsx")