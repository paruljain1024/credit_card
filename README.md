# Card Fraud Analysis
This project implements fraud detection for credit card.

# Steps

# Step 1: Import the Libraries
```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

# Step 2: Read Data
```
transaction_data = pd.read_csv('Creditcard_data.csv')
```

# Step 3: Analyze Dataset Features
```
transaction_data.head()
transaction_data.info()
transaction_data.describe()
```

# Step 4: Check Target Distribution
```
fraud_distribution = transaction_data["Class"].value_counts()
print("Transaction Types:")
print(fraud_distribution)
```

# Step 5: Data Quality Check
```
null_count = transaction_data.isna().sum()
print("Null Values Per Feature:")
print(null_count)
```

# Step 6: Split Transaction Types
```
normal_trans = transaction_data[transaction_data['Class'] == 0]
fraud_trans = transaction_data[transaction_data['Class'] == 1]
print('Normal transactions:', normal_trans.shape)
print('Fraudulent transactions:', fraud_trans.shape)
```

# Step 7: Create Distribution Plot
```
plt.figure(figsize=(10, 5))
fraud_distribution.plot(kind='barh', color='lightblue', 
                       title="Transaction Type Distribution")
plt.xlabel("Count")
plt.ylabel("Class")
```
Visualizes transaction class distribution using horizontal bar plot

# Step 8: Balance Dataset
```
from imblearn.over_sampling import SMOTE
from collections import Counter

target = transaction_data['Class']
features = transaction_data.drop(['Class'], axis=1)

smote_balancer = SMOTE(random_state=42)
features_balanced, target_balanced = smote_balancer.fit_resample(features, target)
```
Uses SMOTE technique to create balanced dataset with equal class distribution

# Step 9: Combine Balanced Data
```
processed_data = pd.concat([
    pd.DataFrame(features_balanced),
    pd.DataFrame(target_balanced, columns=['Class'])
], axis=1)

print("Balanced dataset shape:", processed_data.shape)
print("Class distribution:\n", processed_data['Class'].value_counts())
```

# Step 10: Generate Sample Sets
```
from sklearn.model_selection import train_test_split

# 1. Random Sample
sample1 = processed_data.sample(n=int(0.2 * len(processed_data)), 
                              random_state=42)

# 2. Stratified Sample
grouped = processed_data.groupby('Class')
sample2 = grouped.apply(
    lambda x: x.sample(int(0.2 * len(x)), random_state=42)
).reset_index(drop=True)

# 3. Systematic Sample
interval = len(processed_data) // int(0.2 * len(processed_data))
offset = np.random.randint(0, interval)
sample3 = processed_data.iloc[offset::interval]

# 4. Cluster Sample
n_groups = 5
group_ids = np.arange(len(processed_data)) % n_groups
processed_data['Group'] = group_ids
selected_group = np.random.randint(0, n_groups)
sample4 = processed_data[processed_data['Group'] == selected_group].drop('Group', axis=1)

# 5. Bootstrap Sample
sample5 = processed_data.sample(n=int(0.2 * len(processed_data)), 
                              replace=True, 
                              random_state=42)

print("Sample sizes:", len(sample1), len(sample2), len(sample3), len(sample4), len(sample5))
```

