import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from pprint import pprint


crops = pd.read_csv("soil_measures.csv")


print(crops.isna().sum().sort_values()) # there is no missing values
print(crops["crop"].unique()) # types of crop
X = crops.drop("crop",axis=1)
y = crops["crop"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
feature_score = {}

for feature in X.columns:
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    accuracy = accuracy_score(y_test, y_pred)
    feature_score[feature] = accuracy
pprint(feature_score)
best_feature = max(feature_score, key=feature_score.get)
best_predictive_feature = {best_feature: feature_score[best_feature]}
print("Feature Scores:")
print(feature_score)
print("\nBest Predictive Feature:")
print(best_predictive_feature)
