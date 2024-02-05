import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Step 1: Load the detection result data set
data_path = r'C:\Users\Dell\Desktop\SVM_outlier.csv'
data = pd.read_csv(data_path)

# Separate normal and outlier values
normal_data = data[data['outliers'] == 1]
outliers_data = data[data['outliers'] == -1]

# Step 3: smooth
smoothed_outliers_data = outliers_data.copy()
for column in ['AC', 'DEN']:
 
    smoothed_values = lowess(
        endog=smoothed_outliers_data[column],
        exog=smoothed_outliers_data['DEPTH'],
        frac=0.05
    )
    smoothed_outliers_data[column] = smoothed_values[:, 1]


reconstructed_data = pd.concat([normal_data, smoothed_outliers_data]).sort_values(by='DEPTH')



# Save to CSV file
output_file_path = r'C:\Users\Dell\Desktop\SVM_outlier_Reconstructed_Data.csv'
reconstructed_data.to_csv(output_file_path, index=False)

# Step 5: Save to CSV file
output_file_path = r'C:\Users\Dell\Desktop\SVM_outlier_Reconstructed_Data.csv'
reconstructed_data.to_csv(output_file_path, index=False)

# Determine the maximum and minimum values for the DEPTH column
depth_min = data['DEPTH'].min()
depth_max = data['DEPTH'].max()


# visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) 
plt.plot(data['AC'], data['DEPTH'], color='blue', label='Original AC', linewidth=1.5)
plt.plot(reconstructed_data['AC'], reconstructed_data['DEPTH'], color='green', label='Reconstructed AC', linewidth=1.5)  
plt.ylim(top=1345.9, bottom=1125) 
plt.gca().invert_yaxis() 
plt.ylabel('DEPTH')
plt.xlabel('AC')
#plt.title('AC vs DEPTH')
#plt.legend(loc='upper left') 


plt.subplot(1, 2, 2) 
plt.plot(data['DEN'], data['DEPTH'], color='blue', label='Original DEN', linewidth=1.5) 
plt.plot(reconstructed_data['DEN'], reconstructed_data['DEPTH'], color='green', label='Reconstructed DEN', linewidth=1.5) 
plt.ylim(top=1345.9, bottom=1125) 
plt.gca().invert_yaxis()
plt.ylabel('DEPTH')
plt.xlabel('DEN')
plt.title('DEN vs DEPTH')
plt.legend(loc='upper left') 

plt.tight_layout() 
plt.show()


image_output_path = r'C:\Users\Dell\Desktop\SVM_outlier_Reconstruction_Comparison.png'
plt.savefig(image_output_path, dpi=600) 