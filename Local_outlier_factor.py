import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the data
file_path = 'C:/Users/Dell/Desktop/2/well_log_data_process.csv'
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
# Delete rows with missing values
#data.dropna(subset=['DEPTH', 'AC', 'DEN'], inplace=True)

# Step3：Feature selection
features = data[['AC', 'DEN']]

# Step4：model selection
lof = LocalOutlierFactor()

# Step 5: Model training and prediction
data['outliers'] = lof.fit_predict(features)

# Step 6: Visualize the results
plt.style.use('seaborn-paper')
fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# Draw scatter plots
sns.scatterplot(data=data, x='AC', y='DEPTH', hue='outliers', palette={1: 'blue', -1: 'red'}, ax=axes[0])
sns.scatterplot(data=data, x='DEN', y='DEPTH', hue='outliers', palette={1: 'blue', -1: 'red'}, ax=axes[1])

# Set axis inversion, labels, and legends
for ax in axes:
    ax.invert_yaxis()
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Depth')
    ax.legend(title='Anomaly', labels=['Normal', 'Anomaly'])


plt.savefig('C:/Users/Dell/Desktop/2/LOF_outliers_scatter.png', dpi=300)


fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
sns.histplot(data=data, x='AC', hue='outliers', multiple='stack', palette={1: 'blue', -1: 'red'}, ax=axes[0])
sns.histplot(data=data, x='DEN', hue='outliers', multiple='stack', palette={1: 'blue', -1: 'red'}, ax=axes[1])


for ax in axes:
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Frequency')
    ax.legend(title='Anomaly', labels=['Normal', 'Anomaly'])


plt.savefig('C:/Users/Dell/Desktop/2/LOF_outliers_histograms.png', dpi=300)


output_file_path = 'C:/Users/Dell/Desktop/2/LOF_outlier.csv'
data.to_csv(output_file_path, columns=['DEPTH', 'AC', 'DEN', 'outliers'], index=False)