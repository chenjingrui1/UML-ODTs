import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the data
file_path = 'C:/Users/Dell/Desktop/2/well_log_data_process.csv'
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
# Delete rows with missing values
#data = data.dropna(subset=['DEPTH', 'AC', 'DEN'])

# Step 3: Feature selection
features = data[['AC', 'DEN']]

# Step 4: Model selection & training 
# Initialize Isolation Forest
iso_forest = IsolationForest()

# Step 5: Model training and prediction
iso_forest.fit(features)
scores = iso_forest.decision_function(features)
data['outliers'] = iso_forest.predict(features)
data['scores'] = scores

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
plt.savefig('C:/Users/Dell/Desktop/2/IF_outliers_scatter.png', dpi=300)

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
plt.savefig('C:/Users/Dell/Desktop/2/IF_outliers_histograms.png', dpi=300)

# Save the results with outliers to CSV file
output_file_path = 'C:/Users/Dell/Desktop/2/IF_outlier.csv'
data.to_csv(output_file_path, columns=['DEPTH', 'AC', 'DEN', 'outliers'], index=False)