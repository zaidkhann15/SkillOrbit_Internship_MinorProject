# ============================================================
# Project: End-to-End Exploratory Data Analysis (EDA)
# Dataset Source: Kaggle (Titanic-style Passenger Dataset)
# Author: Thouna | Minor Project Submission
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: DATA AUDIT & CLEANING
# ============================================================


df = pd.read_csv('titanic_data.csv')

print("─── Original Dataset Info ───")
print(f"Shape: {df.shape}")
print(df.dtypes)
print()

# ── Cleaning Logic ──────────────────────────────────────────

# Duplicates
dup_count = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# Missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())    
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  

print("\nMissing values after handling:")
print(df.isnull().sum())

# ============================================================
# STEP 2: OUTLIER DETECTION (IQR Method)
# ============================================================
print("\n─── Outlier Logic ───")

for col in ['Age', 'Fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col}:")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Lower Bound={lower_bound:.2f}, Upper Bound={upper_bound:.2f}")
    print(f"  Outliers detected: {len(outliers)}")

   
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col], patch_artist=True,
                boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                medianprops=dict(color='white', linewidth=2),
                flierprops=dict(marker='o', color='red', markersize=6))
    plt.title(f'Box Plot: {col} (Outliers Visible)')
    plt.tight_layout()
    plt.savefig(f'boxplot_{col.lower()}.png', dpi=120)
    plt.close()

    
    df[col] = df[col].clip(lower_bound, upper_bound)
    print(f"  → Outliers capped to [{lower_bound:.2f}, {upper_bound:.2f}]")

# ============================================================
# STEP 3: ENCODING (Objects to Numerical)
# ============================================================
print("\n─── Encoding Logic ───")
print("dtypes BEFORE encoding:")
print(df[['Gender', 'Embarked']].dtypes)


le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])


df = pd.get_dummies(df, columns=['Embarked'], prefix='Emb')

print("\ndtypes AFTER encoding:")
print(df[['Gender_encoded']].dtypes)
emb_cols = [c for c in df.columns if c.startswith('Emb_')]
print(df[emb_cols].dtypes)
print("\nSample encoded data:")
print(df[['Gender', 'Gender_encoded'] + emb_cols].head())

# ============================================================
# STEP 4: VISUAL ANALYSIS
# ============================================================

sns.set_theme(style='whitegrid')


plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=25, color='#4C72B0', edgecolor='white', alpha=0.85)
plt.axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2,
            label=f"Mean: {df['Age'].mean():.1f}")
plt.axvline(df['Age'].median(), color='green', linestyle='--', linewidth=2,
            label=f"Median: {df['Age'].median():.1f}")
plt.title('Histogram: Age Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('histogram_age.png', dpi=120)
plt.close()


plt.figure(figsize=(8, 5))
colors = df['Survived'].map({0: '#C44E52', 1: '#55A868'})
plt.scatter(df['Age'], df['Fare'], c=colors, alpha=0.65, s=50, edgecolors='white')
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color='#55A868', label='Survived'),
                    Patch(color='#C44E52', label='Did Not Survive')])
plt.title('Scatter Plot: Age vs Fare (by Survival)', fontsize=13, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Fare ($)')
plt.tight_layout()
plt.savefig('scatter_age_fare.png', dpi=120)
plt.close()


num_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Survived', 'Gender_encoded']
corr = df[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1, center=0,
            annot_kws={'size': 10, 'weight': 'bold'})
plt.title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_correlation.png', dpi=120)
plt.close()

# ============================================================
# FINAL SUMMARY — 3 Significant Findings
# ============================================================
print("\n" + "="*55)
print("FINAL SUMMARY — 3 Significant Findings")
print("="*55)

# Finding 1
pclass_surv = df.groupby('Pclass')['Survived'].mean() * 100
print(f"\n1. Class & Survival:")
for cls, rate in pclass_surv.items():
    print(f"   Class {cls}: {rate:.1f}% survival rate")
print("   → Higher class = significantly better survival odds.")

# Finding 2
gender_surv = df.groupby('Gender')['Survived'].mean() * 100
print(f"\n2. Gender & Survival:")
for g, rate in gender_surv.items():
    print(f"   {g}: {rate:.1f}% survival rate")
print("   → Female passengers had noticeably higher survival rates.")

# Finding 3
corr_fare_surv = df['Fare'].corr(df['Survived'])
print(f"\n3. Fare & Survival Correlation: {corr_fare_surv:.3f}")
print("   → Passengers who paid higher fares had better survival chances,")
print("     likely linked to higher passenger class access.")

print("\n✓ Project complete. All charts saved as PNG files.")
