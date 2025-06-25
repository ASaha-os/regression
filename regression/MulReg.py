import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from mpl_toolkits.mplot3d import Axes3D  

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target 

X = df[['MedInc', 'HouseAge']]  
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Plotting 3D Scatter Plot 
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['MedInc'], X_test['HouseAge'], y_test, color='blue', label='Actual', s=20)
ax.scatter(X_test['MedInc'], X_test['HouseAge'], y_pred, color='red', alpha=0.5, label='Predicted', s=20)

ax.set_xlabel('Median Income')
ax.set_ylabel('House Age')
ax.set_zlabel('Target Value')
ax.set_title('Actual vs Predicted (3D View)')
ax.legend()

plt.tight_layout()
plt.show()
