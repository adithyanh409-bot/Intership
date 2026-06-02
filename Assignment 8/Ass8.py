import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("Iris.csv")
data = data.drop("Id", axis=1)

X = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
Y = data["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy =", accuracy*100, "%")
joblib.dump(model, "iris_random_forest_model.joblib")

sample1 = [[5.0, 3.6, 1.4, 0.2]]
print("Input:", sample1[0])
print("Prediction:", model.predict(sample1)[0])

sample2 = [[6.1, 2.8, 4.0, 1.3]]
print("Input:", sample2[0])
print("Prediction:", model.predict(sample2)[0])

sample3 = [[6.9, 3.1, 5.4, 2.1]]
print("Input:", sample3[0])
print("Prediction:", model.predict(sample3)[0])