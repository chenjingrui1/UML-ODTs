import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the data
file_path = 'C:/Users/Dell/Desktop/2/well_log_data_process.csv'
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
# Remove rows with missing values
data.dropna(subset=['DEPTH', 'AC', 'DEN'], inplace=True)

# Step 3: Feature selection
features = data[['AC', 'DEN']]

# Step 4: Model selection
# Initialize the One-Class SVM model with specified nu and gamma
nu_value = 0.05  # outlier fraction
gamma_value = 'auto'  # kernel coefficient
oc_svm = OneClassSVM(nu=nu_value, gamma=gamma_value)

# Step 5: Model training and prediction
oc_svm.fit(features)
predictions = oc_svm.predict(features)

# Adding prediction results back to the original data
data['outliers'] = predictions

# Step 6: Visualization
plt.style.use('seaborn-paper')  # Style for high-quality journal

# Scatter plots for AC and DEN vs DEPTH with normal in blue and outliers in red
fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
sns.scatterplot(data=data, x='AC', y='DEPTH', hue='outliers', palette={1: 'blue', -1: 'red'}, ax=axes[0])
sns.scatterplot(data=data, x='DEN', y='DEPTH', hue='outliers', palette={1: 'blue', -1: 'red'}, ax=axes[1])

# Invert Y-axis for depth and set labels
for ax in axes:
    ax.invert_yaxis()
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Depth')
    ax.legend(title='Anomaly', labels=['Normal', 'Anomaly'])

# Save scatter plots
plt.savefig('C:/Users/Dell/Desktop/2/SVM_outliers_scatter.png', dpi=300)

# Histograms for AC and DEN showing distribution of normal and outliers
fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
sns.histplot(data=data, x='AC', hue='outliers', multiple='stack', palette={1: 'blue', -1: 'red'}, ax=axes[0])
sns.histplot(data=data, x='DEN', hue='outliers', multiple='stack', palette={1: 'blue', -1: 'red'}, ax=axes[1])

# Set labels for histograms
for ax in axes:
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Frequency')
    ax.legend(title='Anomaly', labels=['Normal', 'Anomaly'])

# Save histograms
plt.savefig('C:/Users/Dell/Desktop/2/SVM_outliers_histograms.png', dpi=300)

# Save the results with outliers to CSV file
# output_file_path = 'C:/Users/Dell/Desktop/2/SVM_outlier.csv'
# data.to_csv(output_file_path, columns=['DEPTH', 'AC', 'DEN', 'outliers'], index=False)



# Step 7: Visualization using boxplots
plt.style.use('seaborn-paper')  # Style for high-quality journal

# Boxplots for scores of AC and DEN
fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
sns.boxplot(data=data, x='outliers', y='scores', ax=axes[0])
sns.boxplot(data=data, y='scores', ax=axes[1])

# Set titles and labels
axes[0].set_title('Anomaly Scores by Group')
axes[0].set_xticklabels(['Normal', 'Anomaly'])
axes[0].set_xlabel('Group')
axes[0].set_ylabel('Anomaly Score')

axes[1].set_title('Overall Anomaly Scores')
axes[1].set_xlabel('All Data Points')
axes[1].set_ylabel('Anomaly Score')

# Save boxplots
plt.savefig('C:/Users/Dell/Desktop/2/IF_scores_boxplots.png', dpi=300)

# Save the results with outliers and scores to CSV file
output_file_path = 'C:/Users/Dell/Desktop/2/IF_outlier_scores.csv'
data.to_csv(output_file_path, columns=['DEPTH', 'AC', 'DEN', 'outliers', 'scores'], index=False)