# module9_knn_gridsearchcv.py

"""
User Guide:
-----------
This program implements a k-Nearest Neighbors (kNN) classification system with hyperparameter tuning using GridSearchCV.

Steps:
1. The user is asked to input the number of training samples (N).
2. The user then enters N training pairs (x, y), where:
   - x is a real number (feature),
   - y is a non-negative integer (label/class).
3. The user is asked to input the number of test samples (M).
4. The user enters M test pairs (x, y) in the same format.
5. The program trains a kNN model using the training set.
6. It performs hyperparameter tuning using GridSearchCV to find the best value of k (from 1 to 10).
7. It tests the trained model on the test set and calculates accuracy.
8. It prints the best value of k and the test accuracy.

Dependencies:
-------------
You need the following Python libraries installed:
- numpy
- scikit-learn

You can install them using:
pip install numpy scikit-learn
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def read_pairs(n, set_name):
    """
    Read n pairs of (x, y) from the user input for a given dataset name.

    Parameters:
        n (int): Number of data points to read
        set_name (str): Name of the dataset ("training set" or "test set")

    Returns:
        X (ndarray): Feature values as a column vector of shape (n, 1)
        y (ndarray): Labels as a 1D array of shape (n,)
    """
    X = []
    y = []
    print(f"\nEnter {n} (x, y) pairs for the {set_name}:")
    for i in range(n):
        # Read a real number for feature x
        x_val = float(input(f"  Enter x[{i+1}]: "))
        # Read a non-negative integer for label y
        y_val = int(input(f"  Enter y[{i+1}]: "))
        X.append(x_val)
        y.append(y_val)

    # Reshape X into (n, 1) to fit scikit-learn input format
    return np.array(X).reshape(-1, 1), np.array(y)


def main():
    print("### k-NN Classifier with GridSearchCV ###")

    # Step 1: Read number of training samples
    N = int(input("Enter the number of training pairs (N): "))
    # Step 2: Read N (x, y) pairs for training
    X_train, y_train = read_pairs(N, "training set")

    # Step 3: Read number of test samples
    M = int(input("\nEnter the number of test pairs (M): "))
    # Step 4: Read M (x, y) pairs for testing
    X_test, y_test = read_pairs(M, "test set")

    # Step 5: Initialize kNN classifier
    knn = KNeighborsClassifier()

    # Step 6: Define parameter grid to search over k = 1 to 10
    param_grid = {'n_neighbors': list(range(1, 11))}

    # Step 7: Apply GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)  # Fit using training data

    # Step 8: Extract best k value and best model
    best_k = grid_search.best_params_['n_neighbors']
    best_knn = grid_search.best_estimator_

    # Step 9: Predict labels for the test set using best model
    y_pred = best_knn.predict(X_test)

    # Step 10: Calculate test accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Step 11: Output results
    print("\n=== RESULTS ===")
    print(f"Best k found using GridSearchCV: {best_k}")
    print(f"Test set accuracy: {accuracy:.4f}")

# Python script entry point
if __name__ == "__main__":
    main()
