import pandas as pd
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt

crops = pd.read_csv("soil_measures.csv")

X = crops.drop("crop",axis=1).values
y = crops["crop"].values
le = LabelEncoder()
y = le.fit_transform(y)
names = crops.drop("crop",axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X,y).coef_
plt.bar(names,lasso_coef)
plt.xticks(rotation=45)
plt.show()
feature_index = crops.columns.get_loc("ph")
feature_index2 = crops.columns.get_loc("P")
feature_index3 = crops.columns.get_loc("N")
X[:, feature_index] = X[:, feature_index] * 0.40
X[:, feature_index2] = X[:, feature_index2] * 0.10
X[:, feature_index3] = X[:, feature_index3] * 0.025
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
