import pandas as pd
import numpy as np

#reading the document
india_d_test = pd.read_csv(r'Project Data/India.csv', error_bad_lines=False, infer_datetime_format=True, index_col=False )

#Removing the 'Other' Column from the document
keep_col = ['Date','State','District','Confirmed','Recovered',"Deceased","Tested"]
new_india_d_test = india_d_test [keep_col]

#Creating a new CSV files without the 'Other' column
new_india_d_test.to_csv("india_Test_Cleaned.csv", index=False)
CleanedData = pd.read_csv(r'india_Test_Cleaned.csv', error_bad_lines=False, infer_datetime_format=True)

#Dropping all the rows with a N/A value and creating a new csv file
Ind = CleanedData.dropna(axis = 0, how = 'any')
Ind.to_csv("FinalIndiaData", index=False)
FData = pd.read_csv(r'FinalIndiaData', error_bad_lines=False, infer_datetime_format=True,)

#2021 - 02 - 11 is the most reccuring date

#Attempt to clean the csv file and obtain all the data for the needed date

index_names = FData[FData['Date'] != '2021-02-11'].index

# drop these row indexes
# from dataFrame
FData.drop(index_names, inplace=True)

print(Data)

FData.to_csv(r"FinalDataIndia")







