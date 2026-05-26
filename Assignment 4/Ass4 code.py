import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("machine_failure.csv")

data = data.drop(["UDI","Product ID"], axis=1)

label_encoder = LabelEncoder()
data["Type_encoded"] = label_encoder.fit_transform(data["Type"])

X = data[["Type_encoded","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]",
    "TWF","HDF","PWF","OSF","RNF"]]

Y = data["Machine failure"]

model = LogisticRegression()

model.fit(X, Y)

print("Coeff =", model.coef_)
print("Intercept =", model.intercept_)

type_new_enc = label_encoder.transform(["M"])[0]

print(model.predict([[type_new_enc,301,311,1450,52,28,1,0,0,0,0]]))