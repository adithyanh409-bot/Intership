import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("power_data.csv")

x = data[['Wind_Speed','Blade_Angle','Rotor_Speed']]
y = data[['Power_Output']]

model = LinearRegression()
model.fit(x,y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

wind_speed = float(input("Enter Wind Speed (m/s): "))
blade_angle = float(input("Enter Blade Angle (degrees): "))
rotor_speed = float(input("Enter Rotor Speed (RPM): "))

prediction = model.predict([[wind_speed, blade_angle, rotor_speed]])
print("Predicted Power Output for Wind Speed=", wind_speed, "m/s, Blade Angle=", blade_angle, "degrees, Rotor Speed=", rotor_speed,
"rpm is:", prediction, "kW")