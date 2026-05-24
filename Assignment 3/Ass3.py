import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math

data = pd.read_csv("KNN_Dataset.csv")

x = data[["Temperature"]]
y = data["Fuel_Consumption"]

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x, y)
y_pred = model.predict(x)

mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mse)

# Predict for temperature = 58
prediction = model.predict([[58]])

# Output
print("Predicted Fuel Consumption =", prediction)
print("MSE =", mse)
print("RMSE =", rmse)