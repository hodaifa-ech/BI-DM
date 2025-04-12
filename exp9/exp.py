import numpy as np
import pandas as pd

try:
    dataset = pd.read_csv('Social_Network_Ads.csv')
    print("Dataset loaded successfully.")
    X = dataset.iloc[:, [2, 3]].values 
    y = dataset.iloc[:, -1].values   
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
except FileNotFoundError:
    print("Error: Social_Network_Ads.csv not found. Please ensure it's in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("Feature scaling applied.")

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
print("Gaussian Naive Bayes model trained.")

y_pred = classifier.predict(X_test)
print("Predictions made on the test set.")

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Evaluation Results ---")
print(f"Accuracy Score: {accuracy:.4f} (or {accuracy*100:.2f}%)")
print("\nConfusion Matrix:")
print(cm)
print("\nInterpretation of Confusion Matrix:")
print(f"[[ True Negatives (TN)  False Positives (FP) ]]")
print(f" [ False Negatives (FN) True Positives (TP)  ]]")
tn, fp, fn, tp = cm.ravel() # Flatten the matrix to easily access elements
print(f"\nTN: {tn} (Correctly predicted 'Not Purchased')")
print(f"FP: {fp} (Incorrectly predicted 'Purchased' - Type I Error)")
print(f"FN: {fn} (Incorrectly predicted 'Not Purchased' - Type II Error)")
print(f"TP: {tp} (Correctly predicted 'Purchased')")
print("--- End of Evaluation ---")