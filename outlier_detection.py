import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/content/OUTLIER.CSV')
data.head()

# Scatter Plot of the data
plt.scatter(data['YearsExperience'], data['Salary'], c=data['Outlier'])
plt.show()

# Calculating Z-Score
z_score = ((data['Salary'] - data['Salary'].mean()) / data['Salary'].std())
print(z_score)

# Count of datapoints which have a z-score of greater than 3
print(sum(abs(z_score > 3))) # using the absolute value




