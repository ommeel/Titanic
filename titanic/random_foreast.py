import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score


# Load the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)
test["Fare"].fillna(test["Fare"].mean(), inplace=True)

train = pd.get_dummies(train, columns=["Sex"], drop_first=True)
test = pd.get_dummies(test, columns=["Sex"], drop_first=True)


chosen_features = ["Pclass","Sex_male","Age","Fare"]
X_train = train[chosen_features]
Y_train = train["Survived"]
X_test = test[chosen_features]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

test_passenger_id = test["PassengerId"]
results = pd.DataFrame({
    "PassengerId":test_passenger_id,
    "Survived":Y_predict
})
print(results.head())
results.to_csv('predict_random_foreast.csv', index=False)


# possible_features = ["Age","Pclass","Sex_male", "SibSp","Fare"]
# pack_features = []
# for r in range(1, len(possible_features) + 1):
#     pack_features.extend(itertools.combinations(possible_features, r))

# dic = {}
# pack_features = [list(sub_set) for sub_set in pack_features]
# for features in pack_features:
#     X_train = train[features]
#     Y_train = train["Survived"]

#     model = RandomForestClassifier(n_estimators=50, random_state=42)
#     scores = cross_val_score(model, X_train,Y_train,scoring='accuracy')
#     scores = scores.mean()

#     dic[tuple(features)] = scores

# sorted_features = sorted(dic.items(), key=lambda x:x[1], reverse=True)
# print( "Cross-validation scores:", sorted_features)
