import pandas as pd
import numpy as np

input_file = 'Dataset_GCC_HCM_2015_2025.csv'
df = pd.read_csv(input_file)

df['Date'] = pd.to_datetime(df['Date'])

input_file_2 = 'Dataset_BDS_HCM_Diff.csv'
df2 = pd.read_csv(input_file_2)

df['Price'] = df['Price'].round(4)
df['Interest_Rate'] = df2['Interest_Rate'].round(4)
df['CPI'] = df2['CPI'].round(4)

output_filename = "Dataset_BDS_HCM_Diff.csv"
df.to_csv(output_filename, index=False)
