import pandas as pd

# Read data
file_path = 'C:/Users/Dell/Desktop/2/well_log_data.csv'
data = pd.read_csv(file_path)

# Removes a row that contains a specific missing value (-9999) and a value of 0
cleaned_data = data[(data != -9999) & (data != 0)].dropna()

# Save the processed file
output_path = 'C:/Users/Dell/Desktop/2/well_log_data_process.csv'
cleaned_data.to_csv(output_path, index=False)