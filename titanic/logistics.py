import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# Load the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)

train = pd.get_dummies(train, columns=["Sex"], drop_first=True)
test = pd.get_dummies(test, columns=["Sex"], drop_first=True)
print('test', test.isna().sum())

chosen_features = ["Pclass","Sex_male","SibSp"]
X_train = train[chosen_features]
Y_train = train["Survived"]
X_test = test[chosen_features]

model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

test_passenger_id = test["PassengerId"]
results = pd.DataFrame({
    "PassengerId":test_passenger_id,
    "Survived":Y_predict
})
print(results.head())
results.to_csv('logistics.csv', index=False)


# possible_features = ["Age","Pclass","Sex_male", "SibSp","Parch"]
# pack_features = []
# for r in range(1, len(possible_features)):
#     pack_features.extend(itertools.combinations(possible_features, r))

# dic = {}
# pack_features = [list(sub_set) for sub_set in pack_features]
# for features in pack_features:
#     X_train = train[features]
#     Y_train = train["Survived"]

#     model = LogisticRegression(max_iter=200)
#     scores = cross_val_score(model, X_train,Y_train,scoring='accuracy')
#     scores = scores.mean()

#     dic[tuple(features)] = scores

# sorted_features = sorted(dic.items(), key=lambda x:x[1], reverse=True)
# print("features", features, "Cross-validation scores:", sorted_features)



