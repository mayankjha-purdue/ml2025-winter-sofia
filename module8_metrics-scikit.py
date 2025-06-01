# module8_metrics-scikit.py

# ==========================
# IMPORT REQUIRED LIBRARIES
# ==========================

# Import numpy for efficient array operations and numerical computation
import numpy as np

# Import precision_score and recall_score from sklearn.metrics
# These are standard ML evaluation metrics
from sklearn.metrics import precision_score, recall_score


# ==========================
# MAIN FUNCTION DEFINITION
# ==========================

def main():
    # --------------------------
    # USER INPUT: NUMBER OF POINTS
    # --------------------------
    # Ask the user to enter a positive integer value for N,
    # which represents how many (x, y) data points will be entered.
    N = int(input("Enter the number of (x, y) points: "))

    # --------------------------
    # DATA INITIALIZATION
    # --------------------------
    # Initialize two numpy arrays of size N:
    # X will store the ground truth labels (true class labels)
    # Y will store the predicted labels (model predictions)
    # Both arrays will be initialized with zeros and have integer data type.
    X = np.zeros(N, dtype=int)
    Y = np.zeros(N, dtype=int)

    # --------------------------
    # DATA COLLECTION
    # --------------------------
    # Loop through N times to collect x (true label) and y (predicted label) for each point
    for i in range(N):
        # Prompt user to input the ground truth label (x) for this data point
        # This value must be either 0 or 1
        x = int(input(f"Enter ground truth (x) for point {i + 1} (0 or 1): "))

        # Prompt user to input the predicted label (y) for this data point
        # This value must also be either 0 or 1
        y = int(input(f"Enter predicted value (y) for point {i + 1} (0 or 1): "))

        # Store the values in their respective numpy arrays
        X[i] = x
        Y[i] = y

    # --------------------------
    # METRICS COMPUTATION
    # --------------------------
    # Use scikit-learn's built-in precision_score and recall_score functions
    # to compute the performance of the classifier
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    precision = precision_score(X, Y)
    recall = recall_score(X, Y)

    # --------------------------
    # OUTPUT THE RESULTS
    # --------------------------
    # Print the computed Precision and Recall values rounded to 2 decimal places
    print("\n=== Evaluation Metrics ===")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")


# ==========================
# SCRIPT ENTRY POINT
# ==========================

# Python best practice: this block ensures main() runs
# only when this script is executed directly (not when imported as a module)
if __name__ == "__main__":
    main()
