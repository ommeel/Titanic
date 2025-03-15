import pandas as pd
from autogluon.tabular import TabularPredictor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

predictor = TabularPredictor(label='Survived')

predictor.fit(train)
predictions = predictor.predict(test)

test_passenger_id = test["PassengerId"]
results = pd.DataFrame({
    "PassengerId":test_passenger_id,
    "Survived":predictions
})

results.to_csv("autogluon.csv", index=False)
