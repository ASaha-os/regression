import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.b_0 = 0
        self.b_1 = 0

    def fit(self, x, y):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        n = np.size(x)
        m_x, m_y = np.mean(x), np.mean(y)
        SS_xy = np.sum(y * x) - n * m_y * m_x
        SS_xx = np.sum(x * x) - n * m_x * m_x
        self.b_1 = SS_xy / SS_xx
        self.b_0 = m_y - self.b_1 * m_x

    def predict(self, x):
        x = np.array(x, dtype=float)
        return self.b_0 + self.b_1 * x

    def plot_regression(self, x, y):
        y_pred = self.predict(x)
        plt.scatter(x, y, color='blue', label='Data Points')
        plt.plot(x, y_pred, color='red', label='Regression Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    file_path = input("Enter path to CSV file (with two columns: x and y): ")
    try:
        df = pd.read_csv(file_path)
        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least two columns: x and y")
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values

        model = LinearRegression()
        model.fit(x, y)
        print(f"Coefficients: b_0 = {model.b_0}, b_1 = {model.b_1}")
        model.plot_regression(x, y)
    except Exception as e:
        print(f"Error: {e}")
