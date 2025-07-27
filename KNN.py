###################################
#      K NEAREST NEIGHBOR         #
###################################
# Author: Justin Joon Lee

import os
import glob
import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

directory_path = './data/EMG_data'

all_files = glob.glob(os.path.join(directory_path, "*.csv"))

dataframes = []

for file in all_files:
    df = pd.read_csv(file)
    operation = os.path.basename(file)[0]
    df['operation'] = operation
    dataframes.append(df)

all_data = pd.concat(dataframes, ignore_index=True)

if 'Unnamed: 0' in all_data.columns:
    all_data = all_data.drop(columns=['Unnamed: 0'])

#### Feature Extraction ####

def extract_features(segment):
    """
    Extracts time-domain and frequency-domain features from a segment of EMG data.
    """
    features = {}
    # Time domain features
    features['mean_raw1'] = segment['raw1'].mean()
    features['std_raw1'] = segment['raw1'].std()
    features['max_raw1'] = segment['raw1'].max()
    features['min_raw1'] = segment['raw1'].min()
    features['range_raw1'] = features['max_raw1'] - features['min_raw1']
    features['var_raw1'] = segment['raw1'].var()
    features['median_raw1'] = segment['raw1'].median()

    features['mean_raw2'] = segment['raw2'].mean()
    features['std_raw2'] = segment['raw2'].std()
    features['max_raw2'] = segment['raw2'].max()
    features['min_raw2'] = segment['raw2'].min()
    features['range_raw2'] = features['max_raw2'] - features['min_raw2']
    features['var_raw2'] = segment['raw2'].var()
    features['median_raw2'] = segment['raw2'].median()

    features['mean_raw3'] = segment['raw3'].mean()
    features['std_raw3'] = segment['raw3'].std()
    features['max_raw3'] = segment['raw3'].max()
    features['min_raw3'] = segment['raw3'].min()
    features['range_raw3'] = features['max_raw3'] - features['min_raw3']
    features['var_raw3'] = segment['raw3'].var()
    features['median_raw3'] = segment['raw3'].median()

    features['rms_raw1'] = np.sqrt(np.mean(segment['raw1']**2))
    features['rms_raw2'] = np.sqrt(np.mean(segment['raw2']**2))
    features['rms_raw3'] = np.sqrt(np.mean(segment['raw3']**2))

    features['skewness_raw1'] = segment['raw1'].skew()
    features['skewness_raw2'] = segment['raw2'].skew()
    features['skewness_raw3'] = segment['raw3'].skew()

    features['mad_raw1'] = np.mean(np.abs(segment['raw1'] - segment['raw1'].mean()))
    features['mad_raw2'] = np.mean(np.abs(segment['raw2'] - segment['raw2'].mean()))
    features['mad_raw3'] = np.mean(np.abs(segment['raw3'] - segment['raw3'].mean()))

    # Frequency domain features using FFT
    fft_raw1 = np.fft.fft(segment['raw1'])
    fft_raw2 = np.fft.fft(segment['raw2'])
    fft_raw3 = np.fft.fft(segment['raw3'])

    features['mean_freq_raw1'] = np.mean(np.abs(fft_raw1))
    features['mean_freq_raw2'] = np.mean(np.abs(fft_raw2))
    features['mean_freq_raw3'] = np.mean(np.abs(fft_raw3))

    features['max_freq_raw1'] = np.max(np.abs(fft_raw1))
    features['max_freq_raw2'] = np.max(np.abs(fft_raw2))
    features['max_freq_raw3'] = np.max(np.abs(fft_raw3))

    return features

# Extract features from each segment
feature_list = []
target_list = []

segment_size = 400
for i in range(0, len(all_data), segment_size):
    segment = all_data.iloc[i:i+segment_size]
    if len(segment) < segment_size:
        continue
    features = extract_features(segment)
    feature_list.append(features)
    target_list.append(segment['operation'].iloc[0])

# Create a dataframe from the extracted features and targets
features_df = pd.DataFrame(feature_list)
target_df = pd.Series(target_list)

X = features_df
y = target_df

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters
n_neighbors = 5 # k nearest neighbors
weights = 'distance'
metric = 'euclidean'

#### Training ####

knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

# 1. Evaluate Accuracy
accuracy_knn = accuracy_score(y_test, y_pred)
report_knn = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Accuracy: {accuracy_knn}")
print("KNN Classification Report:")
print(report_knn)

# 2. Generate and display the Confusion Matrix
conf_matrix_knn = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix_knn)

labels = np.asarray([f'{value}' for value in conf_matrix_knn.flatten()]).reshape(conf_matrix_knn.shape)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_knn, annot=labels, fmt="", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix_KNN')
plt.show()