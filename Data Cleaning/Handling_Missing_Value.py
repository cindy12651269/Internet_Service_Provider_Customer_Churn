# -*- coding: utf-8 -*-
"""Handling_missing_values.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mO4sqHGKGKgvSaA9T81A-EZUFIrGa7th
"""

df.info()

# Visualizes distributions of columns with missing values.
import matplotlib.pyplot as plt
import seaborn as sns

# Check the distribution of 'remaining_contract'
sns.histplot(df['remaining_contract'], kde=True)
plt.title('Distribution of Remaining Contract')
plt.show()

# Check the distribution of 'download_avg'
sns.histplot(df['download_avg'], kde=True)
plt.title('Distribution of Download Average')
plt.show()

# Check the distribution of 'upload_avg'
sns.histplot(df['upload_avg'], kde=True)
plt.title('Distribution of Upload Average')
plt.show()

# Fill missing values in 'remaining_contract', 'download_avg', and 'upload_avg' using the median
df['remaining_contract'].fillna(df['remaining_contract'].median(), inplace=True)
df['download_avg'].fillna(df['download_avg'].median(), inplace=True)
df['upload_avg'].fillna(df['upload_avg'].median(), inplace=True)
df.info()
