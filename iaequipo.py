# -*- coding: utf-8 -*-
"""
#Info

- PatientID: A unique identifier assigned to each patient (5034 to 7425).

- Age: The age of the patients ranges from 5 to 80 years.
- Gender: Gender of the patients, where 0 represents Male and 1 represents Female.
- Ethnicity: The ethnicity of the patients, coded as follows:
 - 0: Caucasian
  - 1: African American
  - 2: Asian
  - 3: Other
- EducationLevel: The education level of the patients, coded as follows:
  - 0: None
  - 1: High School
  - 2: Bachelor's
  - 3: Higher
- BMI: Body Mass Index of the patients, ranging from 15 to 40.
- Smoking: Smoking status, where 0 indicates No and 1 indicates Yes.
- PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10.
- DietQuality: Diet quality score, ranging from 0 to 10.
- SleepQuality: Sleep quality score, ranging from 4 to 10.
- PollutionExposure: Exposure to pollution, score from 0 to 10.
- PollenExposure: Exposure to pollen, score from 0 to 10.
- DustExposure: Exposure to dust, score from 0 to 10.
- PetAllergy: Pet allergy status, where 0 indicates No and 1 indicates Yes.
- FamilyHistoryAsthma: Family history of asthma, where 0 indicates No and 1 indicates Yes.
- HistoryOfAllergies: History of allergies, where 0 indicates No and 1 indicates Yes.
- Eczema: Presence of eczema, where 0 indicates No and 1 indicates Yes.
- HayFever: Presence of hay fever, where 0 indicates No and 1 indicates Yes.
- GastroesophagealReflux: Presence of gastroesophageal reflux, where 0 indicates No and 1 indicates Yes.
- LungFunctionFEV1: Forced Expiratory Volume in 1 second (FEV1), ranging from 1.0 to 4.0 liters.
- LungFunctionFVC: Forced Vital Capacity (FVC), ranging from 1.5 to 6.0 liters.
- Wheezing: Presence of wheezing, where 0 indicates No and 1 indicates Yes.
- ShortnessOfBreath: Presence of shortness of breath, where 0 indicates No and 1 indicates Yes.
- ChestTightness: Presence of chest tightness, where 0 indicates No and 1 indicates Yes.
- Coughing: Presence of coughing, where 0 indicates No and 1 indicates Yes.
- NighttimeSymptoms: Presence of nighttime symptoms, where 0 indicates No and 1 indicates Yes.
- ExerciseInduced: Presence of symptoms induced by exercise, where 0 indicates No and 1 indicates Yes.


- Diagnosis: Diagnosis status for Asthma, where 0 indicates No and 1 indicates Yes.


- DoctorInCharge: This column contains confidential information about the doctor in charge, with "Dr_Confid" as the value for all patients.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm

df = pd.read_csv(r'/content/asthma_disease_data.csv')

# Show column names
names = df.columns.tolist()
[print(index, name) for index,name in enumerate(names,1)]

# Show the shape of the DataFrame (rows, columns)
print("DataFrame shape:", df.shape)

df1=df.drop(['PatientID','DoctorInCharge','Gender','Ethnicity','EducationLevel','Smoking','PetAllergy','FamilyHistoryAsthma','HistoryOfAllergies','Eczema','HayFever','GastroesophagealReflux','Wheezing','ShortnessOfBreath','ChestTightness','Coughing','NighttimeSymptoms','ExerciseInduced','Diagnosis'],axis=1)
df1.head()
#Mantener variables double para realizar análisis de correlación y verificar si debemos conservar todas o hay alguna que podamos omitir

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 6), dpi = 150)
paleta = sns.diverging_palette (200, 10,as_cmap=True). reversed ( )
sns. heatmap(df1.corr(), vmin=-1, vmax=1, cmap=paleta,square=True,annot=True,ax=ax)
plt.show

df.head()

diagnosis_counts = df["Diagnosis"].value_counts()
total = len(df)
percentages = diagnosis_counts / total

fig, ax = plt.subplots(figsize=(8, 2))
ax.barh(["Diagnósticos"], percentages[1], color="blue", label="Asma (1)")
ax.barh(["Diagnósticos"], percentages[0], left=percentages[1], color="green", label="No Asma (0)")

ax.set_title("Distribución de Diagnósticos")
ax.set_xlabel("Porcentaje")
ax.legend(loc="center right")


plt.show()

data_cols = ['Age',
 'Gender',
 'Ethnicity',
 'EducationLevel',
 'BMI',
 'Smoking',
 'PhysicalActivity',
 'DietQuality',
 'SleepQuality',
 'PollutionExposure',
 'PollenExposure',
 'DustExposure',
 'PetAllergy',
 'FamilyHistoryAsthma',
 'HistoryOfAllergies',
 'Wheezing',
 'ShortnessOfBreath',
 'ChestTightness',
 'Coughing',
 'NighttimeSymptoms',
 'ExerciseInduced',
 'Diagnosis']
data = df[data_cols].copy()
data.head()

# Check for NA's in all columns
na_counts = data.isna().sum()
print("NA counts for each column:\n", na_counts)

# Histogram for Age, Ethnicity, EducationLevel, BMI, PhysicalActivity, DietQuality, and SleepQuality
hist_cols = ['Age', 'BMI', 'PhysicalActivity', 'DietQuality',
             'SleepQuality','PollenExposure', 'DustExposure', 'PollutionExposure']
data[hist_cols].hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Bar plots for the rest of the variables
bar_cols = ['Ethnicity','Gender', 'Smoking','EducationLevel',
            'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Wheezing',
            'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms',
            'ExerciseInduced', 'Diagnosis']

ncols = 4
nrows = 4

# Create the figure and axes
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 15), layout="constrained")

# Flatten the axs array for easier iteration
axs = axs.flatten()

# Iterate through the bar_cols and plot on each subplot
for i, col in enumerate(bar_cols):
    data[col].value_counts().plot(kind='bar', ax=axs[i], title=col)

# Remove any unused subplots (if num plots < nrows * ncols)
for i in range(len(bar_cols), nrows * ncols):
    fig.delaxes(axs[i])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

len(hist_cols) + len(bar_cols)

"""#Generating Data"""

dataX = data.drop('Diagnosis', axis=1)
dataY = data['Diagnosis']
dataX.head()

dataY.shape,dataX.shape

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state=42 ensures the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42,stratify=dataY)

# Print the shapes of the resulting sets to verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

"""
#Training model
(no correr de nuevo)"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.Input((21,)),
    layers.Dense(30, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(50, activation='relu',),  # Input layer with 21 neurons
    layers.Dropout(0.2),
    layers.Dense(100, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(200, activation='relu',kernel_regularizer=keras.regularizers.l2(0.02)),  # Input layer with 21 neurons
    layers.Dropout(0.4),
    layers.Dense(300, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(200, activation='relu',),  # Input layer with 21 neurons
    layers.Dropout(0.3),
    layers.Dense(100, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(20, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),  # Input layer with 21 neurons
    layers.Dense(8, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron (sigmoid for binary classification)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000)  # You can adjust the number of epochs

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy,'Test loss:',loss)

"""#Pre-trained model"""

# prompt: load keras nn model named model1000.h5

from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('/content/model1000.h5')

# Now you can use the loaded model for predictions or further training
# For example:
# predictions = model.predict(new_data)

"""#Predictions"""

import seaborn as sns
# Make predictions on the train data
y_pred1 = model.predict(X_train)

# Convert predicted probabilities to class labels (0 or 1)
y_pred_classes1 = (y_pred1 >= 0.2).astype(int)

# Create the confusion matrix
cm1 = confusion_matrix(y_train, y_pred_classes1)

print("Confusion Matrix:")
cm1

sns.heatmap(cm1, annot=True, fmt='d').set_title('Confusion matrix of NN with train data')

# Make predictions on the test data
y_pred = model.predict(X_test)

# Convert predicted probabilities to class labels (0 or 1)
y_pred_classes = (y_pred >= 0.2).astype(int)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

print("Confusion Matrix:")
cm

sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of NN with test data')

min(y_pred)

plt.figure(figsize=(10, 6))
plt.hist(y_pred, bins=20, alpha=0.5, label='y_pred', color='blue')
plt.hist(y_test, bins=20, alpha=0.5, label='y_test', color='red')
plt.xlabel('Diagnosis')
plt.ylabel('Frequency')
plt.title('Histogram Comparison of Predicted vs. Actual Diagnosis')
plt.legend(loc='upper right')
plt.show()

"""#SVM Model

"""

# Create an SVM model
svm_model = svm.SVC(kernel='linear', C=1)  # You can experiment with different kernels and C values

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# You can also print a classification report for more detailed evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_svm))

# Create the confusion matrix for the SVM model
cm_svm = confusion_matrix(y_test, y_pred_svm)

print("Confusion Matrix (SVM):")
print(cm_svm)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Make predictions on the training data using the SVM model
y_pred_svm_train = svm_model.predict(X_train)

# Create the confusion matrix for the SVM model with training data
cm_svm_train = confusion_matrix(y_train, y_pred_svm_train)

print("Confusion Matrix (SVM - Training Data):")
print(cm_svm_train)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm_train, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (SVM - Training Data)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

"""#Cross Validation"""

from sklearn.model_selection import cross_val_score, KFold
# Cross-validation for SVM
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Define the cross-validation strategy
svm_scores = cross_val_score(svm_model, dataX, dataY, cv=kf, scoring='accuracy')

print("SVM Cross-validation scores:", svm_scores)
print("SVM Mean accuracy:", svm_scores.mean())

#!pip install scikeras

from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier # Import KerasClassifier
model1 = keras.Sequential([
    layers.Input((21,)),
    layers.Dense(30, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(50, activation='relu',),  # Input layer with 21 neurons
    layers.Dropout(0.2),
    layers.Dense(100, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(200, activation='relu',kernel_regularizer=keras.regularizers.l2(0.02)),  # Input layer with 21 neurons
    layers.Dropout(0.4),
    layers.Dense(300, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(200, activation='relu',),  # Input layer with 21 neurons
    layers.Dropout(0.3),
    layers.Dense(100, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(20, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),  # Input layer with 21 neurons
    layers.Dense(8, activation='relu',),  # Input layer with 21 neurons
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron (sigmoid for binary classification)
])

# Compile the model
model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model = KerasClassifier(model=model1, epochs=100, verbose=0)
# Cross-validation for Neural Network
nn_scores = cross_val_score(model, dataX, dataY, cv=kf, scoring='accuracy')
print("Neural Network Cross-validation scores:", nn_scores)
print("Neural Network Mean accuracy:", nn_scores.mean())

print("Neural Network Cross-validation scores:", nn_scores)
print("Neural Network Mean accuracy:", nn_scores.mean())

"""El svm es ligeramente mejor pero computacionalmente más costoso

#Aprendizaje no supervisado: Clustering
#K - Means vs K - Prototypes
"""

from sklearn.preprocessing import MinMaxScaler

numerical_data = df[hist_cols]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numerical data
scaled_data = scaler.fit_transform(numerical_data)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=hist_cols)

scaled_df.head()

#!pip install kneed

from sklearn.cluster import KMeans
from kneed import KneeLocator

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(f"Elbow method suggests k = {kl.elbow}")

# Plot the elbow curve
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal k")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_df)

# Get cluster labels for each data point
labels = kmeans.labels_

# Add cluster labels to your dataframe
scaled_df['cluster'] = labels

scaled_df.head()

from sklearn.decomposition import PCA

# Select the features for PCA
features = ['BMI', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
            'PollenExposure', 'DustExposure', 'PollutionExposure']
X = scaled_df[features]

# Apply PCA with 1 component
pca = PCA(n_components=1)
pca_result = pca.fit_transform(X)

# Create a new DataFrame with PCA result and age
pca_df = pd.DataFrame({'PCA_1': pca_result.flatten(), 'Age': scaled_df['Age'], 'cluster': scaled_df['cluster']})

# Plot age vs. the first principal component, colored by cluster
plt.figure(figsize=(10, 6))
for cluster in pca_df['cluster'].unique():
    subset = pca_df[pca_df['cluster'] == cluster]
    plt.scatter(subset['Age'], subset['PCA_1'], label=f'Cluster {cluster}')

plt.xlabel('Age')
plt.ylabel('PCA 1')
plt.title('Age vs. PCA 1 (Colored by Cluster)')
plt.legend()
plt.show()

"""#Compare to real labels when k=2"""

kmeans2 = KMeans(n_clusters=2, random_state=42)
kmeans2.fit(scaled_df)

# Get cluster labels for each data point
labels = kmeans2.labels_

# Add cluster labels to your dataframe
scaled_df['cluster'] = labels

scaled_df.head()

comparison_df = pd.DataFrame({'Diagnosis': df['Diagnosis'], 'Cluster': scaled_df['cluster']})
print(comparison_df)

# Further analysis: Contingency table
contingency_table = pd.crosstab(comparison_df['Diagnosis'], comparison_df['Cluster'])
print("\nContingency Table:")
print(contingency_table)

# Calculate and print the accuracy
correct_predictions = np.sum(comparison_df['Diagnosis'] == comparison_df['Cluster'])
total_predictions = len(comparison_df)
accuracy = correct_predictions / total_predictions
print("\nAccuracy:", accuracy)

from sklearn.metrics import recall_score, f1_score

recall = recall_score(comparison_df['Diagnosis'], comparison_df['Cluster'])
print("\nRecall:", recall)

# Calculate F1-score
f1 = f1_score(comparison_df['Diagnosis'], comparison_df['Cluster'])
print("\nF1-score:", f1)

"""##K - Prototypes"""

# !pip install kmodes
# !pip install gower

data.drop(['Diagnosis','Age'], axis=1, inplace=True)
data.head()

bar_cols[:-1]

bar_cols_indices = [data.columns.get_loc(col) for col in bar_cols[:-1] if col in data.columns]
bar_cols_indices

len(bar_cols_indices)+7

from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import gower

# Calculate Gower distance matrix
dist_matrix = gower.gower_matrix(data)

# K-Prototypes with Elbow method and Silhouette analysis
cost = []
silhouette_avg = []
for num_clusters in range(2, 11):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=0)
    kproto.fit_predict(data, categorical=bar_cols_indices)
    cost.append(kproto.cost_)
    cluster_labels = kproto.labels_
    silhouette_avg.append(silhouette_score(dist_matrix, cluster_labels))

# Plot the elbow method
plt.plot(range(2, 11), cost)
plt.xlabel("Number of clusters")
plt.ylabel("Cost")
plt.title("Elbow Method for Optimal k")
plt.show()


# Plot the silhouette analysis
plt.plot(range(2, 11), silhouette_avg)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal k")
plt.show()

kl = KneeLocator(range(2, 11), silhouette_avg, curve="concave", direction="increasing")
optimal_k = kl.elbow
print(f"Optimal k using silhouette: {optimal_k}")

kproto1 = KPrototypes(n_clusters=2, init='Cao', verbose=0)
kproto1.fit_predict(data, categorical=bar_cols_indices)

cluster_labels = kproto1.labels_
data['cluster'] = cluster_labels

data['Age'] = df.copy()['Age']
data.head()

plt.figure(figsize=(10, 6))
for cluster in data['cluster'].unique():
    subset = data[data['cluster'] == cluster]
    plt.scatter(subset['Age'], subset['BMI'], label=f'Cluster {cluster}')

plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Age vs. BMI')
plt.legend()
plt.show()

df2 = data.copy()
df2.drop('cluster', axis=1, inplace=True)

covariance_matrix = df2.cov()

# Set a correlation threshold (e.g., 0.8)
threshold = 0.8

# Find highly correlated variables
highly_correlated_variables = []
for i in range(len(covariance_matrix.columns)):
    for j in range(i + 1, len(covariance_matrix.columns)):
        if abs(covariance_matrix.iloc[i, j]) > threshold:
            highly_correlated_variables.append((covariance_matrix.columns[i],
                                                covariance_matrix.columns[j],
                                                covariance_matrix.iloc[i, j]))

# Print the highly correlated variable pairs and their covariance
print("Highly Correlated Variable Pairs:")
for var1, var2, covariance in highly_correlated_variables:
    print(f"{var1} and {var2}: Covariance = {covariance}")

df2.head()

from sklearn.preprocessing import StandardScaler

pca_cols = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'PhysicalActivity',
            'DietQuality', 'SleepQuality', 'PollenExposure', 'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies',
            'Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing',
            'NighttimeSymptoms', 'ExerciseInduced']


pca_data = data[pca_cols].copy()


categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'PetAllergy',
                   'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Wheezing',
                   'ShortnessOfBreath', 'ChestTightness', 'Coughing',
                   'NighttimeSymptoms', 'ExerciseInduced']

pca_data = pd.get_dummies(pca_data, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
scaled_pca_data = scaler.fit_transform(pca_data)

pca = PCA(n_components=1) # Change n_components as needed
pca_index = pca.fit_transform(scaled_pca_data)

# Add the PCA index to the original DataFrame
data['PCA_Index'] = pca_index

# Print the DataFrame with the PCA index
data[['Age', 'PCA_Index']]

plt.figure(figsize=(10, 6))
for cluster in data['cluster'].unique():
    subset = data[data['cluster'] == cluster]
    plt.scatter(subset['Age'], subset['PCA_Index'], label=f'Cluster {cluster}')

plt.xlabel('Age')
plt.ylabel('PCA Index')
plt.title('Age vs. PCA Index')
plt.legend()
plt.show()