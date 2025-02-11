# -*- coding: utf-8 -*-
"""Jointplot_correlation_exploration.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mO4sqHGKGKgvSaA9T81A-EZUFIrGa7th
"""

#Jointplot for Exploring Joint Distributions
# Jointplot for 'download_avg' and 'upload_avg'
sns.jointplot(x='download_avg', y='upload_avg', data=df, kind='scatter', height=8)
plt.show()
# Low value clustering: Most users have low download and upload averages, concentrated in the bottom-left corner.
# Positive correlation: As download averages increase, upload averages also rise, though the correlation weakens at higher values.

# Jointplot for 'download_avg' and 'bill_avg'
sns.jointplot(x='download_avg', y='bill_avg', data=df, kind='scatter', height=8)
plt.show()
# Low value concentration: Most users have low download and upload averages, as shown by the cluster in the lower left.
# Outliers: A few users have higher averages, but they are less common.