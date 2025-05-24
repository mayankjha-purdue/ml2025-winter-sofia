import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import NotFittedError

def main():
    try:
        # Step 1: Ask the user to input the number of data points (N)
        # This defines how many (x, y) pairs the user will provide
        N = int(input("Enter the number of data points (N): "))
        if N <= 0:
            raise ValueError("N must be a positive integer.")

        # Step 2: Ask the user to input the number of neighbors (k) for k-NN regression
        k = int(input("Enter the number of neighbors (k): "))
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # Step 3: Validate that k is not more than N
        if k > N:
            print("Error: k cannot be greater than N.")
            return  # Early exit due to invalid input

        # Step 4: Initialize data containers using NumPy
        # X_data is a 2D array with shape (N, 1), representing the input features
        # y_data is a 1D array with shape (N,), representing the labels/targets
        print(f"\nEnter {N} (x, y) data points:")
        X_data = np.zeros((N, 1))  # Allocate space for feature values
        y_data = np.zeros(N)       # Allocate space for label values

        # Step 5: Read and store each (x, y) pair from the user
        for i in range(N):
            x = float(input(f"Enter x[{i+1}]: "))  # Read the x-coordinate (feature)
            y = float(input(f"Enter y[{i+1}]: "))  # Read the y-coordinate (target)
            X_data[i] = x
            y_data[i] = y

        # Step 6: Compute and display the variance of the y values
        # Variance shows how much the labels vary around their mean
        variance = np.var(y_data)
        print(f"\nVariance of the target labels (y): {variance:.4f}")

        # Step 7: Ask the user to enter a new input value X for prediction
        # This is the query point at which we want to predict Y using k-NN regression
        X_query = float(input("\nEnter the X value to predict Y using k-NN: "))
        X_query = np.array([[X_query]])  # Reshape to 2D array for scikit-learn compatibility

        # Step 8: Create and train the k-NN Regressor model using scikit-learn
        # The model is trained on the full (X, y) dataset
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_data, y_data)  # Fit the model

        # Step 9: Use the trained model to predict the Y value for the query X
        prediction = model.predict(X_query)

        # Step 10: Output the predicted result
        print(f"\nPredicted Y for X = {X_query[0][0]}: {prediction[0]:.4f}")

    except ValueError as e:
        # Handles non-numeric or invalid user inputs
        print(f"Input error: {e}")
    except NotFittedError:
        # Catches errors related to prediction without fitting the model
        print("Error: Model was not trained properly.")
    except Exception as e:
        # General error handling for any unexpected issues
        print(f"An unexpected error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
