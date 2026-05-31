# %% [markdown]
# # Eksperimen Preprocessing - Dataset Iris
# Notebook ini mengikuti Template Eksperimen MSML
# Nama Siswa: Nama-siswa

# %% [markdown]
# ## 1. Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %% [markdown]
# ## 2. Data Loading

# %%
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Shape dataset:", df.shape)
df.head()

# %%
# Simpan raw dataset
df.to_csv("iris_raw.csv", index=False)
print("Raw dataset disimpan ke iris_raw.csv")

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
# Informasi dasar dataset
print("=== INFO DATASET ===")
df.info()

# %%
# Statistik deskriptif
print("=== STATISTIK DESKRIPTIF ===")
df.describe()

# %%
# Cek missing values
print("=== MISSING VALUES ===")
print(df.isnull().sum())

# %%
# Cek duplikasi
print("=== DUPLIKASI ===")
print(f"Jumlah baris duplikat: {df.duplicated().sum()}")

# %%
# Distribusi kelas
print("=== DISTRIBUSI KELAS ===")
print(df['species'].value_counts())

# %%
# Visualisasi distribusi kelas
plt.figure(figsize=(6, 4))
df['species'].value_counts().plot(kind='bar', color=['steelblue', 'salmon', 'lightgreen'])
plt.title('Distribusi Kelas')
plt.xlabel('Species')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_class_distribution.png')
plt.show()

# %%
# Visualisasi boxplot untuk deteksi outlier
feature_cols = ['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)']

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for i, col in enumerate(feature_cols):
    axes[i].boxplot(df[col])
    axes[i].set_title(col)
fig.suptitle('Boxplot Fitur (Sebelum Penghapusan Outlier)')
plt.tight_layout()
plt.savefig('eda_boxplot.png')
plt.show()

# %%
# Heatmap korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(df[feature_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antar Fitur')
plt.tight_layout()
plt.savefig('eda_correlation.png')
plt.show()

# %%
# Pairplot
sns.pairplot(df, hue='species', vars=feature_cols)
plt.savefig('eda_pairplot.png')
plt.show()

# %% [markdown]
# ## 4. Preprocessing

# %% [markdown]
# ### 4.1 Handling Missing Values

# %%
missing = df.isnull().sum()
print("Missing values per kolom:")
print(missing)

if missing.sum() == 0:
    print("\nTidak ada missing values. Data sudah bersih.")
else:
    df = df.dropna()
    print(f"\nMissing values dihapus. Shape setelah: {df.shape}")

# %% [markdown]
# ### 4.2 Handling Duplicates

# %%
dup = df.duplicated().sum()
print(f"Jumlah duplikasi: {dup}")

if dup > 0:
    df = df.drop_duplicates()
    print(f"Duplikasi dihapus. Shape setelah: {df.shape}")
else:
    print("Tidak ada duplikasi.")

# %% [markdown]
# ### 4.3 Handling Outliers (IQR Method)

# %%
print(f"Shape sebelum hapus outlier: {df.shape}")

for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print(f"Shape setelah hapus outlier: {df.shape}")

# %% [markdown]
# ### 4.4 Label Encoding

# %%
le = LabelEncoder()
df['target'] = le.fit_transform(df['species'])
print("Label encoding selesai.")
print(f"Classes: {list(le.classes_)}")
print(df['target'].value_counts())

# %% [markdown]
# ### 4.5 Train-Test Split

# %%
X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# %% [markdown]
# ### 4.6 Feature Scaling

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling (StandardScaler) selesai.")
print(f"Mean X_train (setelah scaling): {X_train_scaled.mean(axis=0).round(4)}")
print(f"Std X_train (setelah scaling): {X_train_scaled.std(axis=0).round(4)}")

# %% [markdown]
# ### 4.7 Simpan Hasil Preprocessing

# %%
import os
os.makedirs("iris_preprocessing", exist_ok=True)

train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
train_df['target'] = y_train.values
train_df.to_csv("iris_preprocessing/iris_train.csv", index=False)

test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
test_df['target'] = y_test.values
test_df.to_csv("iris_preprocessing/iris_test.csv", index=False)

print("Data preprocessed disimpan ke folder iris_preprocessing/")
print(f"Train: iris_preprocessing/iris_train.csv ({train_df.shape})")
print(f"Test:  iris_preprocessing/iris_test.csv ({test_df.shape})")

# %%
# Verifikasi hasil
train_check = pd.read_csv("iris_preprocessing/iris_train.csv")
test_check = pd.read_csv("iris_preprocessing/iris_test.csv")
print("\nVerifikasi train dataset:")
print(train_check.head())
print("\nVerifikasi test dataset:")
print(test_check.head())
