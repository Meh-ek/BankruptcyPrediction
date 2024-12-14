# BankruptcyPrediction
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils import resample
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

path = '/content/drive/My Drive/Machine Learning Project/'

data = pd.read_csv(path + 'data.csv')

# Handle missing values
data = data.dropna()

# Define features and target
X = data.drop(columns=['Bankrupt?'])  # Replace 'Bankruptcy' with actual target column
y = data['Bankrupt?']  # Replace with actual target column

# Train-test split (stratified to maintain class balance in the test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Logistic Regression
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred_baseline = baseline_model.predict(X_test_scaled)
print("Confusion Matrix (Baseline Logistic Regression):\n", confusion_matrix(y_test, y_pred_baseline))
print("\nClassification Report (Baseline Logistic Regression):\n", classification_report(y_test, y_pred_baseline))


# Handle missing values
data = data.dropna()  # Drop rows with missing values

# Define features and target
X = data.drop(columns=['Bankrupt?'])  # Replace 'Bankruptcy' with the actual target column
y = data['Bankrupt?']  # Replace with the actual target column

# Create interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interaction = poly.fit_transform(X)

# Standardize features (Scaling AFTER interaction terms are generated)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_interaction)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model with L1 regularization
model = LogisticRegression(max_iter=1000, penalty='l1', solver='saga', random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Display coefficients of the logistic regression model
feature_names = poly.get_feature_names_out(input_features=X.columns)
coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("\nTop Features with Coefficients:\n", coefficients.head(10))


# Ensure train-test split is stratified and test set is untouched
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Separate training data into majority and minority classes
majority_train = X_train[y_train == 0]
minority_train = X_train[y_train == 1]

# Perform clustering on minority class
kmeans = KMeans(n_clusters=5, random_state=42).fit(minority_train)
minority_train['Cluster'] = kmeans.labels_

# Resample each cluster
resampled_minority = pd.concat([
    resample(
        minority_train[minority_train['Cluster'] == i].drop(columns=['Cluster']),
        replace=True,
        n_samples=len(majority_train) // 5,
        random_state=42
    ) for i in range(5)
])

# Combine resampled minority with majority
balanced_train = pd.concat([majority_train, resampled_minority])
X_balanced = balanced_train
y_balanced = pd.concat([
    pd.Series(0, index=majority_train.index),
    pd.Series(1, index=resampled_minority.index)
])

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_balanced, y_balanced)

# Evaluate on the untouched test set
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



import tensorflow as tf

# Define Neural Network model
nn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

# Evaluate on test set
loss, accuracy = nn_model.evaluate(X_test_scaled, y_test)
print(f"Neural Network Test Accuracy: {accuracy:.2f}")

# Generate predictions
y_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype(int)
print("Confusion Matrix (Neural Network):\n", confusion_matrix(y_test, y_pred_nn))
print("\nClassification Report (Neural Network):\n", classification_report(y_test, y_pred_nn))



