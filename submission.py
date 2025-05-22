import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost as xgb
from xgboost import XGBClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
# Set random seed untuk reproducibility
np.random.seed(42)


df = pd.read_csv('animal_disease_dataset.csv')

df.head()

df.columns = df.columns.str.replace(' ', '_')

df.info()

df.describe()

df.describe(include='object')

missing_values = df.isna().sum()
print("Missing Values:")
print(missing_values)

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y='Age', data=df)
plt.title('Age')
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
sns.boxplot(y='Temperature', data=df)
plt.title('Temperature')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Animal', y='Age', data=df)
plt.title('Age berdasarkan Animal')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
sns.boxplot(x='Animal', y='Temperature', data=df)
plt.title('Temperature berdasarkan Animal')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

df_z = df.copy()
for col in ['Age', 'Temperature']:
    col_zscore = col + '_zscore'
    df_z[col_zscore] = stats.zscore(df_z[col])
    
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(df_z.index, df_z['Age_zscore'], alpha=0.5)
plt.axhline(y=3, color='r', linestyle='-', alpha=0.3, label='Threshold (z=3)')
plt.axhline(y=-3, color='r', linestyle='-', alpha=0.3)
plt.title('Z-Score Age')
plt.ylabel('Z-Score')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(df_z.index, df_z['Temperature_zscore'], alpha=0.5)
plt.axhline(y=3, color='r', linestyle='-', alpha=0.3, label='Threshold (z=3)')
plt.axhline(y=-3, color='r', linestyle='-', alpha=0.3)
plt.title('Z-Score Temperature')
plt.ylabel('Z-Score')
plt.legend()
plt.tight_layout()
plt.show()
age_outliers = len(df_z[abs(df_z['Age_zscore']) > 3])
temp_outliers = len(df_z[abs(df_z['Temperature_zscore']) > 3])
print(f"Jumlah outliers dalam Age: {age_outliers}")
print(f"Jumlah outliers dalam Temperature: {temp_outliers}")

cat_features = df.select_dtypes(include=['object']).columns
num_features = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(16, 10))
animal_counts = df['Animal'].value_counts()
animal_percentages = 100 * df['Animal'].value_counts(normalize=True).round(2)
animal_counts.plot(kind='bar', title='Distribution of Animal')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
for i, p in enumerate(animal_percentages):
    plt.annotate(f"{p}%", (i, animal_counts.iloc[i]), ha='center', va='bottom')

plt.figure(figsize=(16, 10))
symptom1_counts = df['Symptom_1'].value_counts()
symptom1_percentages = 100 * df['Symptom_1'].value_counts(normalize=True).round(2)
if len(symptom1_counts) > 10:
    symptom1_counts = symptom1_counts.head(10)
symptom1_counts.plot(kind='bar', title='Distribusi Symptom_1')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
for i, p in enumerate(symptom1_percentages[:len(symptom1_counts)]):
    plt.annotate(f"{p}%", (i, symptom1_counts.iloc[i]), ha='center', va='bottom')

plt.figure(figsize=(16, 10))
symptom2_counts = df['Symptom_2'].value_counts()
symptom2_percentages = 100 * df['Symptom_2'].value_counts(normalize=True).round(2)
if len(symptom2_counts) > 10:
    symptom2_counts = symptom2_counts.head(10)
symptom2_counts.plot(kind='bar', title='Distribusi Symptom_2')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
for i, p in enumerate(symptom2_percentages[:len(symptom2_counts)]):
    plt.annotate(f"{p}%", (i, symptom2_counts.iloc[i]), ha='center', va='bottom')

plt.figure(figsize=(16, 10))
symptom3_counts = df['Symptom_3'].value_counts()
symptom3_percentages = 100 * df['Symptom_3'].value_counts(normalize=True).round(2)
if len(symptom3_counts) > 10:
    symptom3_counts = symptom3_counts.head(10)
symptom3_counts.plot(kind='bar', title='Distribution of Symptom_3')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
for i, p in enumerate(symptom3_percentages[:len(symptom3_counts)]):
    plt.annotate(f"{p}%", (i, symptom3_counts.iloc[i]), ha='center', va='bottom')

plt.figure(figsize=(16, 10))
disease_counts = df['Disease'].value_counts()
disease_percentages = 100 * df['Disease'].value_counts(normalize=True).round(2)
if len(disease_counts) > 10:
    disease_counts = disease_counts.head(10)
disease_counts.plot(kind='bar', title='Distribution of Disease')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
for i, p in enumerate(disease_percentages[:len(disease_counts)]):
    plt.annotate(f"{p}%", (i, disease_counts.iloc[i]), ha='center', va='bottom')


for feature in ['Animal', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Disease']:
    counts = df[feature].value_counts()
    percentages = 100 * df[feature].value_counts(normalize=True).round(2)
    summary = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    
    print(f"\n--- {feature} ---")
    if len(summary) > 15:
        print(summary.head(15))
        print(f"... and {len(summary)-15} more categories")
    else:
        print(summary)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
age_bins = np.arange(df['Age'].min() - 0.5, df['Age'].max() + 1.5, 1)
sns.histplot(df['Age'], bins=age_bins, kde=True)
plt.title('Histogram Age', fontsize=14)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
temp_bins = np.arange(df['Temperature'].min() - 0.1, df['Temperature'].max() + 0.1, 0.2)
sns.histplot(df['Temperature'], bins=temp_bins, kde=True)
plt.title('Histogram Temperature', fontsize=14)
plt.xlabel('Temperature (°F)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
animal_disease = pd.crosstab(df['Animal'], df['Disease'], normalize='index')
sns.heatmap(animal_disease, annot=True, cmap='YlGnBu', fmt='.2%', linewidths=.5)
plt.title('Distribusi Disease bedasarkan Animal Type', fontsize=16)
plt.ylabel('Animal')
plt.xlabel('Disease')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 8))
pd.crosstab(df['Animal'], df['Disease'], normalize='index').plot(kind='bar', stacked=True)
plt.title('Distribusi Disease bedasarkan Animal Type', fontsize=16)
plt.xlabel('Animal Type')
plt.ylabel('Percentage')
plt.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 15))
plt.subplot(3, 1, 1)
symptom1_disease = pd.crosstab(df['Disease'], df['Symptom_1'], normalize='index')
sns.heatmap(symptom1_disease, cmap='YlGnBu', annot=False)
plt.title('Distribusi Symptom 1 bedasarkan Disease', fontsize=14)
plt.ylabel('Disease')
plt.xlabel('Symptom 1')
plt.subplot(3, 1, 2)
symptom2_disease = pd.crosstab(df['Disease'], df['Symptom_2'], normalize='index')
sns.heatmap(symptom2_disease, cmap='YlGnBu', annot=False)
plt.title('Distribusi Symptom 2 bedasarkan Disease', fontsize=14)
plt.ylabel('Disease')
plt.xlabel('Symptom 2')
plt.subplot(3, 1, 3)
symptom3_disease = pd.crosstab(df['Disease'], df['Symptom_3'], normalize='index')
sns.heatmap(symptom3_disease, cmap='YlGnBu', annot=False)
plt.title('Distribusi Symptom 1 bedasarkan Disease', fontsize=14)
plt.ylabel('Disease')
plt.xlabel('Symptom 3')
plt.tight_layout()
plt.show()

def display_top_symptoms(disease_name, top_n=3):
    disease_data = df[df['Disease'] == disease_name]
    
    top_symptom1 = disease_data['Symptom_1'].value_counts().nlargest(top_n)
    top_symptom2 = disease_data['Symptom_2'].value_counts().nlargest(top_n)
    top_symptom3 = disease_data['Symptom_3'].value_counts().nlargest(top_n)
    
    print(f"Top symptoms untuk {disease_name}:")
    print(f"Top Symptom 1: {', '.join([f'{s} ({top_symptom1[s]/len(disease_data):.1%})' for s in top_symptom1.index])}")
    print(f"Top Symptom 2: {', '.join([f'{s} ({top_symptom2[s]/len(disease_data):.1%})' for s in top_symptom2.index])}")
    print(f"Top Symptom 3: {', '.join([f'{s} ({top_symptom3[s]/len(disease_data):.1%})' for s in top_symptom3.index])}")
    print()
for disease in df['Disease'].unique():
    display_top_symptoms(disease)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='Disease', y='Age', data=df)
plt.title('Distribusi Age bedasarkan Disease', fontsize=14)
plt.xlabel('Disease')
plt.ylabel('Age (years)')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.boxplot(x='Disease', y='Temperature', data=df)
plt.title('Distribusi Temperature bedasarkan Disease', fontsize=14)
plt.xlabel('Disease')
plt.ylabel('Temperature (°F)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.violinplot(x='Disease', y='Age', data=df)
plt.title('Distribusi Kepadatan Age bedasarkan Disease', fontsize=14)
plt.xlabel('Disease')
plt.ylabel('Age (years)')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.violinplot(x='Disease', y='Temperature', data=df)
plt.title('Distribusi Kepadatan Temperature bedasarkan Disease', fontsize=14)
plt.xlabel('Disease')
plt.ylabel('Temperature (°F)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


disease_animal_stats = df.groupby(['Disease', 'Animal']).agg({
    'Age': 'mean',
    'Temperature': 'mean'
}).reset_index()
age_pivot = disease_animal_stats.pivot(index='Disease', columns='Animal', values='Age')
temp_pivot = disease_animal_stats.pivot(index='Disease', columns='Animal', values='Temperature')
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
sns.heatmap(age_pivot, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Rata-rata Age bedasarkan Disease dan Animal Type', fontsize=14)
plt.subplot(1, 2, 2)
sns.heatmap(temp_pivot, annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Rata-rata Temperature bedasarkan Disease dan Animal Type', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
numeric_corr = df[['Age', 'Temperature']].corr()
sns.heatmap(numeric_corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix - Fitur Numerik', fontsize=16)
plt.tight_layout()
plt.show()
df_encoded = df.copy()
# Label encoding untuk target variable (Disease)
le = LabelEncoder()
df_encoded['Disease_encoded'] = le.fit_transform(df_encoded['Disease'])
# One-hot encoding untuk Animal
animal_dummies = pd.get_dummies(df_encoded['Animal'], prefix='Animal')
df_encoded = pd.concat([df_encoded, animal_dummies], axis=1)
# Menggunakan pendekatan frequency encoding untuk gejala
symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3']
# Membuat dictionary untuk memetakan setiap gejala dengan frekuensinya
symptom_freq = {}
for col in symptom_cols:
    for symptom, count in df_encoded[col].value_counts().items():
        if symptom in symptom_freq:
            symptom_freq[symptom] += count
        else:
            symptom_freq[symptom] = count
top_symptoms = sorted(symptom_freq.items(), key=lambda x: x[1], reverse=True)[:24]
for symptom, _ in top_symptoms:
    df_encoded[f'has_{symptom.replace(" ", "_")}'] = df_encoded[symptom_cols].apply(
        lambda row: int(symptom in row.values), axis=1
    )
features_for_corr = ['Age', 'Temperature', 'Disease_encoded'] + \
                    list(animal_dummies.columns) + \
                    [col for col in df_encoded.columns if col.startswith('has_')]
corr_matrix = df_encoded[features_for_corr].corr()
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, 
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix - All Features', fontsize=18)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
target_corr = corr_matrix['Disease_encoded'].sort_values(ascending=False)
plt.figure(figsize=(12, 8))
target_corr.drop('Disease_encoded').plot(kind='barh')
plt.title('Korelasi Feature dengan Disease', fontsize=16)
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

le_disease = LabelEncoder()
df['Disease_encoded'] = le_disease.fit_transform(df['Disease'])
disease_mapping = dict(zip(le_disease.classes_, le_disease.transform(le_disease.classes_)))
print("Mapping Disease ke nilai numerik:")
for disease, code in disease_mapping.items():
    print(f"{disease}: {code}")
    
df[['Disease', 'Disease_encoded']].drop_duplicates().sort_values('Disease_encoded')

animal_dummies = pd.get_dummies(df['Animal'], prefix='Animal')
print("Hasil One-Hot Encoding untuk Animal:")
animal_dummies.head()

symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3']
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].unique())
print(f"Total gejala unik: {len(all_symptoms)}")
print(f"Contoh gejala: {list(all_symptoms)[:5]}")

# has_symptom = 1 jika hewan memiliki gejala tersebut, 0 jika tidak
symptom_features = pd.DataFrame()
for symptom in all_symptoms:
    feature_name = f"has_{symptom.replace(' ', '_')}"
    symptom_features[feature_name] = df[symptom_cols].apply(
        lambda row: int(symptom in row.values), axis=1
    )
print(f"Dimensi fitur gejala: {symptom_features.shape}")
symptom_features.head()


respiratory_symptoms = ['shortness of breath', 'chest discomfort', 'coughing', 'rapid breathing']
df['respiratory_syndrome'] = df[symptom_cols].apply(
    lambda row: sum(1 for symptom in row if symptom in respiratory_symptoms), axis=1
)
systemic_symptoms = ['fatigue', 'sweats', 'chills', 'fever', 'depression', 'loss of appetite']
df['systemic_syndrome'] = df[symptom_cols].apply(
    lambda row: sum(1 for symptom in row if symptom in systemic_symptoms), axis=1
)
foot_mouth_symptoms = ['difficulty walking', 'lameness', 'blisters', 'sores', 'drooling']
df['foot_mouth_syndrome'] = df[symptom_cols].apply(
    lambda row: sum(1 for symptom in row if symptom in foot_mouth_symptoms), axis=1
)
lumps_symptoms = ['painless lumps', 'painful lumps', 'swelling in neck', 
                  'swelling in extremities', 'swelling in abdomen']
df['lumps_syndrome'] = df[symptom_cols].apply(
    lambda row: sum(1 for symptom in row if symptom in lumps_symptoms), axis=1
)
plt.figure(figsize=(15, 10))
for i, syndrome in enumerate(['respiratory_syndrome', 'systemic_syndrome', 
                             'foot_mouth_syndrome', 'lumps_syndrome']):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=syndrome, hue='Disease', data=df)
    plt.title(f'Distribusi {syndrome} berdasarkan Disease')
    plt.xlabel('Jumlah gejala')
    plt.xticks(rotation=0)
    if i == 1 or i == 3:
        plt.legend([])
    else:
        plt.legend(title='Disease', loc='upper right')
plt.tight_layout()
plt.show()

prepared_features = pd.concat([
    df[['Age', 'Temperature', 'respiratory_syndrome', 'systemic_syndrome', 'foot_mouth_syndrome', 'lumps_syndrome']],
    animal_dummies,
    symptom_features
], axis=1)
print(f"Dimensi data dengan semua fitur: {prepared_features.shape}")

X = prepared_features
y = df['Disease_encoded']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
plt.figure(figsize=(12, 10))
mi_scores.head(20).plot.barh()
plt.title('Top 20 Fitur Berdasarkan Mutual Information Score')
plt.xlabel('Mutual Information Score')
plt.tight_layout()
plt.show()

chi2_scores, p_values = chi2(X, y)
chi2_scores = pd.Series(chi2_scores, index=X.columns)
p_values = pd.Series(p_values, index=X.columns)
feature_selection_df = pd.DataFrame({
    'chi2_score': chi2_scores,
    'p_value': p_values,
    'mi_score': mi_scores
}).sort_values('chi2_score', ascending=False)
plt.figure(figsize=(12, 10))
feature_selection_df['chi2_score'].head(20).plot.barh()
plt.title('Top 20 Fitur Berdasarkan Chi-Square Score')
plt.xlabel('Chi-Square Score')
plt.tight_layout()
plt.show()
print("Top 20 Fitur Berdasarkan Chi-Square:")
print(feature_selection_df.head(20))

significant_features = feature_selection_df[feature_selection_df['p_value'] < 0.05].index.tolist()
print(f"Jumlah fitur yang signifikan: {len(significant_features)}")
top_features = mi_scores.sort_values(ascending=False).head(30).index.tolist()
print(f"Top 30 fitur berdasarkan mutual information: {len(top_features)}")
selected_syndrome_features = ['respiratory_syndrome', 'systemic_syndrome', 
                             'foot_mouth_syndrome', 'lumps_syndrome', 
                             'Age', 'Temperature']
final_features = list(set(top_features + selected_syndrome_features))
print(f"Jumlah fitur final: {len(final_features)}")
print("Contoh fitur yang dipilih:", final_features[:10])
X_selected = X[final_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.25, random_state=42, stratify=y
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("\nDistribusi kelas di training set:")
print(pd.Series(y_train).value_counts(normalize=True).round(3) * 100)
print("\nDistribusi kelas di testing set:")
print(pd.Series(y_test).value_counts(normalize=True).round(3) * 100)

numeric_features = ['Age', 'Temperature']
numeric_features = [f for f in numeric_features if f in final_features]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
if numeric_features:
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
if numeric_features:
    comparison = pd.DataFrame({
        'Before_Scaling': X_train[numeric_features].mean(),
        'After_Scaling': X_train_scaled[numeric_features].mean()
    })
    print("Perbandingan rata-rata sebelum dan sesudah scaling:")
    print(comparison)
    
    comparison_std = pd.DataFrame({
        'Before_Scaling': X_train[numeric_features].std(),
        'After_Scaling': X_train_scaled[numeric_features].std()
    })
    print("\nPerbandingan standar deviasi sebelum dan sesudah scaling:")
    print(comparison_std)

class_distribution = pd.Series(y_train).value_counts()
print("Distribusi kelas sebelum resampling:")
print(class_distribution)
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar')
plt.title('Distribusi Kelas Penyakit Sebelum Resampling')
plt.ylabel('Jumlah Sampel')
plt.grid(True, alpha=0.3)
plt.show()
imbalance_ratio = class_distribution.max() / class_distribution.min()
print(f"\nRasio ketidakseimbangan (max/min): {imbalance_ratio:.2f}")
if imbalance_ratio > 1.5:
    print("Terdapat ketidakseimbangan kelas, menerapkan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    resampled_distribution = pd.Series(y_train_resampled).value_counts()
    print("\nDistribusi kelas setelah resampling:")
    print(resampled_distribution)
    
    plt.figure(figsize=(10, 6))
    resampled_distribution.plot(kind='bar')
    plt.title('Distribusi Kelas Penyakit Setelah Resampling')
    plt.ylabel('Jumlah Sampel')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    X_train_final = X_train_resampled
    y_train_final = y_train_resampled
else:
    print("Distribusi kelas cukup seimbang, tidak perlu resampling.")
    X_train_final = X_train_scaled
    y_train_final = y_train

print(f"Missing values di X_train_final: {X_train_final.isna().sum().sum()}")
print(f"Missing values di X_test_scaled: {X_test_scaled.isna().sum().sum()}")
print(f"\nDimensi akhir X_train_final: {X_train_final.shape}")
print(f"Dimensi akhir y_train_final: {y_train_final.shape}")
print(f"Dimensi akhir X_test_scaled: {X_test_scaled.shape}")
print(f"Dimensi akhir y_test: {y_test.shape}")
if numeric_features:
    numeric_summary = pd.DataFrame({
        'Train_Min': X_train_final[numeric_features].min(),
        'Train_Max': X_train_final[numeric_features].max(),
        'Test_Min': X_test_scaled[numeric_features].min(),
        'Test_Max': X_test_scaled[numeric_features].max()
    })
    print("\nRange nilai fitur numerik:")
    print(numeric_summary)
feature_names = list(X_train_final.columns)
print(f"\nFeature names yang akan digunakan ({len(feature_names)} fitur):")
print(feature_names[:10], "...")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    n_classes = len(np.unique(y_test))
    roc_auc = {}
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr, tpr)
    
    avg_roc_auc = sum(roc_auc.values()) / len(roc_auc)
    
    print(f"\n===== {model_name} Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average ROC AUC: {avg_roc_auc:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=le_disease.classes_)
    print(report)
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_disease.classes_, 
                yticklabels=le_disease.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': avg_roc_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics

xgb_model = XGBClassifier(
    objective='multi:softprob',
    n_estimators=100,          
    learning_rate=0.1,         
    max_depth=5,               
    min_child_weight=1,        
    gamma=0,                   
    subsample=0.8,             
    colsample_bytree=0.8,      
    random_state=42            
)
print("Training XGBoost model...")
start_time = datetime.now()
xgb_model.fit(X_train_final, y_train_final)
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()
print(f"XGBoost training completed in {training_time:.2f} seconds")
xgb_metrics = evaluate_model(xgb_model, X_train_final, y_train_final, 
                             X_test_scaled, y_test, "XGBoost")

rf_model = RandomForestClassifier(
    n_estimators=100,      
    max_depth=None,        
    min_samples_split=2,   
    min_samples_leaf=1,    
    max_features='sqrt',   
    bootstrap=True,        
    random_state=42,       
    n_jobs=-1              
)
print("Training Random Forest model...")
start_time = datetime.now()
rf_model.fit(X_train_final, y_train_final)
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()
print(f"Random Forest training completed in {training_time:.2f} seconds")
rf_metrics = evaluate_model(rf_model, X_train_final, y_train_final, 
                           X_test_scaled, y_test, "Random Forest")

models_comparison = pd.DataFrame({
    'XGBoost': [
        xgb_metrics['accuracy'],
        xgb_metrics['precision'],
        xgb_metrics['recall'],
        xgb_metrics['f1_score'],
        xgb_metrics['roc_auc']
    ],
    'Random Forest': [
        rf_metrics['accuracy'],
        rf_metrics['precision'],
        rf_metrics['recall'],
        rf_metrics['f1_score'],
        rf_metrics['roc_auc']
    ]
}, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
print("Model Performance Comparison:")
print(models_comparison)
plt.figure(figsize=(12, 6))
models_comparison.plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
for i in range(len(models_comparison)):
    for j in range(len(models_comparison.columns)):
        plt.text(i + j/len(models_comparison.columns) - 0.2, 
                 models_comparison.iloc[i, j] + 0.01,
                 f'{models_comparison.iloc[i, j]:.3f}',
                 ha='center')
plt.tight_layout()
plt.show()
from sklearn.metrics import classification_report
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)
report_xgb = classification_report(y_test, y_pred_xgb, 
                                  target_names=le_disease.classes_, 
                                  output_dict=True)
report_rf = classification_report(y_test, y_pred_rf, 
                                 target_names=le_disease.classes_, 
                                 output_dict=True)
per_class_comparison = []
for disease in le_disease.classes_:
    per_class_comparison.append({
        'Disease': disease,
        'XGBoost Precision': report_xgb[disease]['precision'],
        'RF Precision': report_rf[disease]['precision'],
        'XGBoost Recall': report_xgb[disease]['recall'],
        'RF Recall': report_rf[disease]['recall'],
        'XGBoost F1': report_xgb[disease]['f1-score'],
        'RF F1': report_rf[disease]['f1-score']
    })
per_class_df = pd.DataFrame(per_class_comparison)
per_class_df = per_class_df.set_index('Disease')
print("\nPer-Class Performance Comparison:")
print(per_class_df)
plt.figure(figsize=(12, 6))
per_class_df[['XGBoost F1', 'RF F1']].plot(kind='bar')
plt.title('F1 Score Comparison by Disease')
plt.ylabel('F1 Score')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()