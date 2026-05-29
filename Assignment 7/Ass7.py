import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv("Decision_Tree _2.csv")

X = data[["tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"]]
Y = data["fail"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X,Y)

prediction = model.predict([[4,5,3,6,1,45,5,1]])

if prediction[0] == 1:
    print(prediction,"The machine is likely to fail")
else:
    print(prediction,"The machine is not likely to fail")

plt.figure(figsize=(12,8))
tree.plot_tree(model,feature_names=["tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"],class_names=["No Failure","Failure"],
               filled=True)
plt.title("Decision Tree Classifier")
plt.show()