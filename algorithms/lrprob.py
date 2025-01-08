from .utils import *
from sklearn.linear_model import LogisticRegression

def train(train_features, train_labels):
    classifier = LogisticRegression(random_state = 42, max_iter=2000, verbose=1)
    classifier.fit(train_features, train_labels)
    return classifier

def get_preds(model, test_features):
    predictions = model.predict(test_features)
    return predictions