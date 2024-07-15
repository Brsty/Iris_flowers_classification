#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Define column names based on dataset description
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Load the dataset
iris_df = pd.read_csv('iris.data', header=None, names=column_names)

# Display the first few rows of the DataFrame to verify the data
print(iris_df.head())


# In[4]:


# Check for missing values
missing_values = iris_df.isnull().sum()
print("\nMissing values:\n", missing_values)

# Summary statistics
summary_statistics = iris_df.describe()
print("\nSummary statistics:\n", summary_statistics)


# In[5]:


# Set up the figure for visualizations
plt.figure(figsize=(12, 10))

# Plot distributions of each feature
plt.subplot(2, 2, 1)
sns.histplot(iris_df['sepal_length'], kde=True)
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)
sns.histplot(iris_df['sepal_width'], kde=True)
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)
sns.histplot(iris_df['petal_length'], kde=True)
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)
sns.histplot(iris_df['petal_width'], kde=True)
plt.title('Petal Width Distribution')

plt.tight_layout()
plt.show()



# In[ ]:




