import numpy as np


class KNearestNeighborRegressor:
    def __init__(self, num_neighbors):
        self.k = num_neighbors
        self.points = None

    def load_data(self, point_list):
        self.points = np.array(point_list, dtype=np.float64)

    def estimate(self, target_x):
        if self.points is None or self.points.shape[0] < self.k:
            raise RuntimeError("Insufficient data: k must be <= number of samples.")

        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        distances = np.abs(x_coords - target_x)
        nearest = np.argsort(distances)[:self.k]
        return np.mean(y_coords[nearest])


def get_positive_integer(prompt_text):
    while True:
        try:
            value = int(input(prompt_text))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Try again.")


def collect_data(n_points):
    entries = []
    print(f"\nProvide {n_points} points in the form of x and y values:")
    for idx in range(1, n_points + 1):
        while True:
            try:
                x_val = float(input(f"  x#{idx}: "))
                y_val = float(input(f"  y#{idx}: "))
                entries.append((x_val, y_val))
                break
            except ValueError:
                print("Invalid coordinates. Please enter numbers.")
    return entries


def main():
    n = get_positive_integer("How many data points (N)? ")
    k = get_positive_integer("Choose k (number of neighbors): ")

    data = collect_data(n)

    try:
        query_x = float(input("\nEnter a value of X to estimate Y: "))
    except ValueError:
        print("Invalid input for X.")
        return

    model = KNearestNeighborRegressor(num_neighbors=k)
    model.load_data(data)

    try:
        result = model.estimate(query_x)
        print(f"\nEstimated Y at X = {query_x}: {round(result, 4)}")
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
