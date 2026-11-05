
import pandas as pd
from sklearn.datasets import load_iris
from statistics import mode
# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("First 5 rows of dataset:")
print(df.head())
# Mean (average)
mean_values = df.mean()
print("\nMean values:")
print(mean_values)
# Median (middle value)
median_values = df.median()
print("\nMedian values:")
print(median_values)
# Mode (most frequent value)
# Pandas mode returns a DataFrame (can be multiple modes)
mode_values = df.mode().iloc[0]
print("\nMode values:")
print(mode_values)

import pandas as pd
from sklearn.datasets import load_iris
# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("First 5 rows of dataset:")
print(df.head())
# Range (max - min)
range_values = df.max() - df.min()
print("\nRange values:")
print(range_values)
# Variance
variance_values = df.var()
print("\nVariance values:")
print(variance_values)
# Standard Deviation
std_values = df.std()
print("\nStandard Deviation values:")
print(std_values)
# Interquartile Range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print("\nInterquartile Range (IQR):")
print(IQR)

import pandas as pd
from sklearn.datasets import load_iris
# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("First 5 rows of dataset:")
print(df.head())
# Skewness (asymmetry of distribution)
skewness_values = df.skew()
print("\nSkewness values:")
print(skewness_values)
# Kurtosis (peakedness / tail heaviness)
kurtosis_values = df.kurt()
print("\nKurtosis values:")
print(kurtosis_values)