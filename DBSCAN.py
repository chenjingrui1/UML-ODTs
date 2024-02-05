import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1:Read the data
file_path = 'C:/Users/Dell/Desktop/2/well_log_data_process.csv'
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
#data.dropna(subset=['DEPTH', 'AC', 'DEN'], inplace=True)

#  Step 3: Feature selection
features = data[['AC', 'DEN']]

# Step 4: Model selection & training 
dbscan = DBSCAN(eps=0.5, min_samples=10)

# Step 5: Model training and prediction
data['outliers'] = dbscan.fit_predict(features)
data['outliers'] = data['outliers'].apply(lambda x: 1 if x != -1 else -1)

# Step 6: Visualization
plt.style.use('seaborn-paper')
fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)


sns.scatterplot(data=data, x='AC', y='DEPTH', hue='outliers', palette={1: 'blue', -1: 'red'}, ax=axes[0])
sns.scatterplot(data=data, x='DEN', y='DEPTH', hue='outliers', palette={1: 'blue', -1: 'red'}, ax=axes[1])


for ax in axes:
    ax.invert_yaxis()
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Depth')
    ax.legend(title='Anomaly', labels=['Normal', 'Anomaly'])


plt.savefig('C:/Users/Dell/Desktop/2/DBSCAN_outliers_scatter.png', dpi=300)


fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
sns.histplot(data=data, x='AC', hue='outliers', multiple='stack', palette={1: 'blue', -1: 'red'}, ax=axes[0])
sns.histplot(data=data, x='DEN', hue='outliers', multiple='stack', palette={1: 'blue', -1: 'red'}, ax=axes[1])


for ax in axes:
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Frequency')
    ax.legend(title='Anomaly', labels=['Normal', 'Anomaly'])


plt.savefig('C:/Users/Dell/Desktop/2/DBSCAN_outliers_histograms.png', dpi=300)


output_file_path = 'C:/Users/Dell/Desktop/2/DBSCAN_outlier.csv'
data.to_csv(output_file_path, columns=['DEPTH', 'AC', 'DEN', 'outliers'], index=False)