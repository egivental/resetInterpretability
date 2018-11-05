from model.Model import Model
from sklearn.tree import DecisionTreeClassifier as SKLearn_DT

class DecisionTree(Model):
    def __init__(self):
        Model.__init__(self)
        self.classifier = SKLearn_DT()
        self.name = "DecisionTree"

    def fit(self, X, Y):
        self.classifier = self.classifier.fit(X, Y)
        return self.classifier

    def predict(self, X):
        return self.classifier.predict(X)
