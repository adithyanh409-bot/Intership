import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv("Load_Extension-Sheet1.csv")

x = data[["Load (N)"]]
y = data[["Extension (mm)"]]

plt.scatter(x,y)
plt.show()
model = LinearRegression()
model.fit(x,y)

new_load = int(input("Enter the new load in N:")) #new_load=55 N
Predict = model.predict([[new_load]])
print("Predicted extension for a load of",new_load,"N is",Predict,"mm")