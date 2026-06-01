# %% [markdown]
# # Eksperimen Preprocessing - Dataset Wine Quality
# Nama Siswa: Reinhart Jens Robert

# %% [markdown]
# ## 1. Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 2. Data Loading

# %%
df = pd.read_csv("../../winequality_raw.csv", sep=';')
print("Shape dataset:", df.shape)
df.head()

# %%
print("Kolom dataset:", df.columns.tolist())
print("\nTipe data:\n", df.dtypes)

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
print("=== INFO DATASET ===")
df.info()

# %%
print("=== STATISTIK DESKRIPTIF ===")
df.describe()

# %%
print("=== MISSING VALUES ===")
print(df.isnull().sum())

# %%
print("=== DUPLIKASI ===")
print(f"Jumlah baris duplikat: {df.duplicated().sum()}")

# %%
print("=== DISTRIBUSI TARGET (quality) ===")
print(df['quality'].value_counts())

# %%
plt.figure(figsize=(8, 4))
df['quality'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Distribusi Kelas Quality')
plt.xlabel('Quality Score')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_class_distribution.png')
plt.show()

# %%
feature_cols = [col for col in df.columns if col != 'quality']
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].boxplot(df[col])
    axes[i].set_title(col)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Boxplot Fitur (Sebelum Penghapusan Outlier)')
plt.tight_layout()
plt.savefig('eda_boxplot.png')
plt.show()

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df[feature_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antar Fitur')
plt.tight_layout()
plt.savefig('eda_correlation.png')
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
    print("\nTidak ada missing values.")
else:
    df = df.dropna()
    print(f"\nMissing values dihapus. Shape: {df.shape}")

# %% [markdown]
# ### 4.2 Handling Duplicates

# %%
dup = df.duplicated().sum()
print(f"Jumlah duplikasi: {dup}")
if dup > 0:
    df = df.drop_duplicates()
    print(f"Duplikasi dihapus. Shape: {df.shape}")
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
# ### 4.4 Binarize Target

# %%
df['quality'] = (df['quality'] >= 6).astype(int)
print("Target setelah binarisasi:")
print(df['quality'].value_counts())

# %% [markdown]
# ### 4.5 Train-Test Split

# %%
X = df[feature_cols]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# %% [markdown]
# ### 4.6 Feature Scaling

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling selesai.")

# %% [markdown]
# ### 4.7 Simpan Hasil Preprocessing

# %%
import os
os.makedirs("winequality_preprocessing", exist_ok=True)

train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
train_df['quality'] = y_train.values
train_df.to_csv("winequality_preprocessing/winequality_train.csv", index=False)

test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
test_df['quality'] = y_test.values
test_df.to_csv("winequality_preprocessing/winequality_test.csv", index=False)

print("Data preprocessing disimpan ke folder winequality_preprocessing/")
print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")

# %%
train_check = pd.read_csv("winequality_preprocessing/winequality_train.csv")
print("\nVerifikasi train dataset:")
print(train_check.head())