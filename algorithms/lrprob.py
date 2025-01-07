from .utils import *
from sklearn.linear_model import LogisticRegression

def train(train_features, train_labels):
    model = LogisticRegression(random_state = 42, max_iter=2000, verbose=1)
    model.fit(train_features, train_labels)
    return model

def get_pred(model, test_features):
    predictions = model.predict(test_features)
    return predictions