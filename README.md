### 1.Topic:Detection of logging and geophysical data and reconstruction methods using unsupervised machine learning outlier detection techniques (UML-ODTs)
This study aimed to investigate the utilization of unsupervised machine learning outlier detection techniques (UML-ODTs) to identify and reconstruct outliers in oil and gas logging curves and to assess their impact on the quality of seismic synthetic records.


### 2. Installation
Via PyPI
Such as:
```bash
pip install sklearn
```
```bash
pip install matplotlib
```
      

## 3.Instructions
Step 1:Data preprocessing(The code file name is data_process.py)
Depending on the data, run this code if preprocessing is required, and if there is no null or zero value, then you don't need to run this code.
```python
import pandas as pd

# Read data
file_path = 'C:/Users/Dell/Desktop/2/well_log_data.csv'
data = pd.read_csv(file_path)

# Removes a row that contains a specific missing value (-9999) and a value of 0
cleaned_data = data[(data != -9999) & (data != 0)].dropna()

# Save the processed file
output_path = 'C:/Users/Dell/Desktop/2/well_log_data_process.csv'
cleaned_data.to_csv(output_path, index=False)
```
Step 2:Choosing the UML-ODTS algorithm to detect outliers in logging curves
(The code file name is DBSCAN.py, IF.py, Local_outlier_factor.py, One-class SVM.py)
```python
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
```
```python
from sklearn.ensemble import IsolationForest
```
```python
from sklearn.neighbors import LocalOutlierFactor
```
```python
from sklearn.svm import OneClassSVM
```
Step 3:Reconstructing outliers in logging curves(The code file name is Reconstructed_Data.py)
```python
from statsmodels.nonparametric.smoothers_lowess import lowess
```



## Contribution

We welcome all forms of contribution. Check out 'CONTRIBUTING.md' to learn how to get started.

## Licence

MIT

## Contact information

If you have any questions, please contact me through 782701032@qq.com.
```
