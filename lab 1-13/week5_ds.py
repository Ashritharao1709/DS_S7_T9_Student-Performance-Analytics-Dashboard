
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Example dataset
data = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [5, 15, 25, 35, 45]
})

# Apply Min-Max Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Convert back to DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

print("Normalized Data (MinMaxScaler):")
print(normalized_df)

#Z-score normalization(Standardization)
import pandas as pd

from sklearn.preprocessing import StandardScaler
# Example dataset
data = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [5, 15, 25, 35, 45]
})


# Apply Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)#fit and transform the data

# Convert back to DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=data.columns)

print("\nStandardized Data (Z-score):")
print(standardized_df)

df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue']})

# One-hot encoding
one_hot = pd.get_dummies(df, columns=['Color'])

print("\nOne-Hot Encoding:")
print(one_hot)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import seaborn as sns
# Load dataset
tips = sns.load_dataset("tips")   # or pd.read_csv("tips.csv")
print("Original Data (first 5 rows):")
print(tips.head())

# 1. Normalization (Min-Max Scaling)
numeric_cols = tips.select_dtypes(include=['float64','int64']).columns #Selects only numeric columns from the dataset('total_bill', 'tip', 'size').

scaler_minmax = MinMaxScaler()  #Create MinMaxScaler Object
tips_normalized = tips.copy()  #A copy is created to preserve the original data.
tips_normalized[numeric_cols] = scaler_minmax.fit_transform(tips[numeric_cols])
print("\nNormalized Data (first 5 rows):")
print(tips_normalized.head())
# 2. Standardization (Z-score)
scaler_standard = StandardScaler()
tips_standardized = tips.copy()
tips_standardized[numeric_cols] = scaler_standard.fit_transform(tips[numeric_cols])

print("\nStandardized Data (first 5 rows):")
print(tips_standardized.head())
# 3. Encoding Categorical Variables
# (a) One-Hot Encoding
tips_onehot = pd.get_dummies(tips, columns=['sex','smoker','day','time'])
print("\nOne-Hot Encoded Data (first 5 rows):")
print(tips_onehot.head())

# Principal Component Analysis (PCA)
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
tips = sns.load_dataset("tips")

# Select numeric columns
numeric_cols = tips.select_dtypes(include=['float64','int64'])  # Select only columns whose data type is float64 or int64

# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_cols)
# Step 2: Apply PCA
pca = PCA(n_components=2)   # project data into 2 dimensions
pca_result = pca.fit_transform(scaled_data)
# Step 3: Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1','PC2'])
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("\nPCA Result (first 5 rows):")
print(pca_df.head())

# Linear Discriminant Analysis (LDA)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: Load dataset
iris = load_iris()
X = iris.data        # features
y = iris.target      # target labels

# Step 2: Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)  # reduce to 2 dimensions
X_lda = lda.fit_transform(X, y)

# Step 3: Create DataFrame with LDA results
lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
lda_df['target'] = y

print("Explained variance ratio:", lda.explained_variance_ratio_)
print("\nLDA Result (first 5 rows):")
print(lda_df.head())

# Step 4: Visualize LDA projection
plt.figure(figsize=(8,6))
for label, color in zip([0,1,2], ['red','green','blue']):
    plt.scatter(
        lda_df.loc[lda_df['target']==label, 'LD1'],
        lda_df.loc[lda_df['target']==label, 'LD2'],
        label=iris.target_names[label],
        color=color,
        alpha=0.7
    )

plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.title('LDA Projection of Iris Dataset')
plt.legend()
plt.show()

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()

X = iris.data        # features
y = iris.target      # target labels

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    max_iter=1000
)

X_tsne = tsne.fit_transform(X)

# Create DataFrame with t-SNE results
tsne_df = pd.DataFrame(
    X_tsne,
    columns=['TSNE1', 'TSNE2']
)

tsne_df['target'] = y

# Display first 5 rows
print("t-SNE Result (first 5 rows):")
print(tsne_df.head())

# Visualize t-SNE projection
plt.figure(figsize=(8, 6))

for label, color in zip(
    [0, 1, 2],
    ['red', 'green', 'blue']
):

    plt.scatter(
        tsne_df.loc[tsne_df['target'] == label, 'TSNE1'],
        tsne_df.loc[tsne_df['target'] == label, 'TSNE2'],
        label=iris.target_names[label],
        color=color,
        alpha=0.7
    )

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Projection of Iris Dataset')
plt.legend()
plt.show()