
import numpy as np
from scipy.spatial import distance

# Define points
pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

# Euclidean distance
euclidean_distance = distance.euclidean(pointA, pointB)
print("Euclidean Distance:", euclidean_distance)

# Similarity (using 1 / (1 + distance))
similarity_euclidean = 1 / (1 + euclidean_distance)
print("Similarity (Euclidean):", similarity_euclidean)

#3.similarity and dissimilarity measures for data analysis
#3.1vector based distance measures(Manhattan Distance)
import numpy as np
from scipy.spatial import distance

# Define points
pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

# Manhattan distance
manhattan_distance = distance.cityblock(pointA, pointB)
print("Manhattan Distance:", manhattan_distance)

#similarity(inverse of distance)
similarity_manhattan = 1 / (1 + manhattan_distance)
print("Manhattan Similarity:", similarity_manhattan)

#3.similarity and dissimilarity measures for data analysis
#3.1vector based distance measures(Minkowski Distance)
import numpy as np
from scipy.spatial import distance

# Define points
pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

# Minkowski distance (example with p=3)
minkowski_distance = distance.minkowski(pointA, pointB, p=3)
print("Minkowski Distance:", minkowski_distance)

# Minkowski similarity (inverse of distance)
minkowski_similarity = 1 / (1 + minkowski_distance)
print("Minkowski Similarity:", minkowski_similarity)

#3.similarity and dissimilarity measures for data analysis
#3.2.Statistical Measures (pearson correlation)
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'X': [10, 20, 30, 40, 50],
    'Y': [12, 24, 33, 45, 60]
})

# Pearson correlation matrix
corr_matrix = df.corr(method='pearson')
print("Pearson Correlation Matrix:\n", corr_matrix)

#For Dataset
print("Pearson Correlations Dataset:\n")
df = pd.read_csv("pearson_correlation_dataset.csv")
pearson_corr = df.corr(method='pearson')
print(pearson_corr)

#3.2.Statistical Measures(Spearman Rank)
import pandas as pd
from scipy.stats import spearmanr

# Create dataset directly
df = pd.DataFrame({
    'X': [10, 20, 30, 40, 50],
    'Y': [12, 18, 33, 47, 55]
})

corr_value, p_value = spearmanr(df['X'], df['Y'])
print("\nSpearman Correlation Coefficient (X vs Y):", corr_value)
print("Spearman Correlation p-value (X vs Y):", p_value)

def hamming_distance(str1, str2):
  #ensure strings are of equal length
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")
    #Count differing positions
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

# Example usage
s1 = "karolin"
s2 = "kathrin"

dist = hamming_distance(s1, s2)
print(f"Hamming distance between {s1} and {s2}: {dist}")

def jaccard_index(str1, str2):
    set1,set2 = set(str1.split()),set(str2.split())
    intersection = (set1.intersection(set2))
    union =(set1.union(set2))
    return len(intersection) / len(union)

# Example usage with texts
s1 = "data science is fun"
s2 = "science makes data useful"

print("Jaccard Index:", jaccard_index(s1, s2))