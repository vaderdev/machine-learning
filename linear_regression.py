import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("data/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "failures", "studytime", "absences"]]

predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

def train(num: int):
    best = 0
    for _ in range(num):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        print(acc)

        if acc > best:
            best = acc
            with open("student.model", "wb") as f:
                pickle.dump(linear,f)
        print("Best: " , best)

model_file = open("student.model", "rb")
linear = pickle.load(model_file)

# y = mx + b
print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'studytime'
style.use("ggplot")
plt.scatter(data[p],data['G3'])
plt.ylabel('Final Grade')
plt.xlabel(p)
plt.show()