import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

data = pd.read_csv("Decision_Tree.csv")

label_encoder = LabelEncoder()

data["Temperature_encoded"] = label_encoder.fit_transform(data["Temperature"])
data["Vibration_encoded"] = label_encoder.fit_transform(data["Vibration"])
data["Failure_encoded"] = label_encoder.fit_transform(data["Failure"])

X = data[["Temperature_encoded","Vibration_encoded"]]
Y = data["Failure_encoded"]

model = DecisionTreeClassifier(criterion="entropy")

model.fit(X, Y)

temp_new = label_encoder.fit(["High","Low","Medium"]).transform(["High"])[0]
vib_new = label_encoder.fit(["High","Low","Medium"]).transform(["Medium"])[0]

prediction = model.predict([[temp_new, vib_new]])

if prediction[0] == 1:
    print(prediction,"The machine is likely to fail")
else:    
    print(prediction,"The machine is not likely to fail")
tree.plot_tree(model)