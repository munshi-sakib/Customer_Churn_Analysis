# Python-Coustomer_Churn-Analysis
.
.
.

![Customer Churn Analysis](https://drive.google.com/uc?export=view&id=179i8OvqQyjLf2c4TqTledS9c4lqx-8om)

## Project Screenshots

### 1. Customer Churn Analysis — Screenshot 1
![Customer Churn Analysis 1](https://drive.google.com/uc?export=view&id=1iaN4WcI6AffnEeEudXZ6SYJew8sO-zXa)

### 2. Customer Churn Analysis — Screenshot 2
![Customer Churn Analysis 2](https://drive.google.com/uc?export=view&id=13NlRFBB3eBfylRqJt3PLVyiXqd5XMWnY)

### 3. Customer Churn Analysis — Screenshot 3
![Customer Churn Analysis 3](https://drive.google.com/uc?export=view&id=1Uc_QTIul4JbxOnYKOmwM4x0GYAFUoU_f)

### 4. Customer Churn Analysis — Screenshot 4

![Customer Churn Analysis 4](https://drive.google.com/uc?export=view&id=1dJTfLxWRZqyNmP5qbIK-Y6MwcYm-lV4P)


``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Project View

# ==========================================
# 1. Data Loading and Cleaning
# ==========================================
df = pd.read_csv('customer churn.csv')
print("Dataset Shape:", df.shape)
df.info()

# Handle blank values in 'TotalCharges' and convert to float
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0").astype(float)
print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("Duplicate Customer IDs:", df["customerID"].duplicated().sum())
```

``` python
# ==========================================
# 2. Target Variable Analysis (Churn)
# Analytical Question: What is the overall percentage and distribution of churned vs. retained customers?
# ==========================================
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Churn", data=df)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Distribution of Churn")
plt.show()

plt.figure(figsize=(3, 3))
gb = df.groupby("Churn").agg({'Churn': "count"})
plt.pie(gb['Churn'], labels=gb.index, autopct="%.1f%%")
plt.title("Percentage of Churned Customers")
plt.show()

# ==========================================
# 3. Demographic Analysis (Gender & Senior Citizen)
# Analytical Question: Does customer churn vary based on demographics like gender or senior citizen status?
# ==========================================
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="gender", data=df, hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Gender")
plt.show()

ct = pd.crosstab(df["SeniorCitizen"], df["Churn"])
ct_percent = ct.div(ct.sum(axis=1), axis=0) * 100
ax = ct_percent.plot(kind="bar", stacked=True, figsize=(7, 5))
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", label_type="center")
plt.title("Churn Percentage by Senior Citizen Status")
plt.xlabel("Senior Citizen")
plt.ylabel("Percentage")
plt.legend(title="Churn")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.show()

# ==========================================
# 4. Customer Tenure Analysis
# Analytical Question: How does customer tenure affect their likelihood of churning?
# ==========================================
plt.figure(figsize=(12, 6))
sns.histplot(x="tenure", data=df, bins=72, hue="Churn", multiple="stack")
plt.title("Customer Churn Distribution by Tenure")
plt.show()

# ==========================================
# 5. Contract Type & Payment Method Analysis
# Analytical Question: How do contract types and payment methods influence customer retention?
# ==========================================
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Contract", data=df, hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Contract Type")
plt.show()

plt.figure(figsize=(8, 5))
ax = sns.countplot(x="PaymentMethod", data=df, hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Payment Method")
plt.xticks(rotation=45)
plt.show()

# ==========================================
# 6. Comprehensive Categorical Feature Grid
# Analytical Question: Across all service and account features, which categories show the most pronounced differences?
# ==========================================
cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
for ax, col in zip(axes.flat, cols):
    sns.countplot(data=df, x=col, ax=ax, hue='Churn')
    ax.set_title(f'{col} vs Churn')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
```
