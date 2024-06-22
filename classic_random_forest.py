import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

class ClassicRandomForest:
    def __init__(self, n_classes, classes, n_estimators=100, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        self.n_classes = n_classes
        self.classes = classes

    def fit(self, X, y):
        N = len(y)
        self.w = np.full(N, 1/N)



        while len(self.trees) != self.n_estimators:
            bootstrap_indices = np.random.choice(N, size=N, replace=True, p=self.w)
            X_bs = []
            y_bs = []
            for idx in bootstrap_indices:
                X_bs.append(X[idx].copy())
                y_bs.append(y[idx].copy())
            X_bootsrtap, y_bootstrap = np.array(X_bs), np.array(y_bs)

            tree = DecisionTreeClassifier(max_features=self.max_features, criterion='entropy')
            tree.fit(X_bootsrtap, y_bootstrap)

            self.trees.append(tree)

    def predict(self, X):
        # Suma ważonych predykcji z poszczególnych drzew
        pred_sum = np.zeros((X.shape[0], len(self.trees[0].classes_)))

        for tree in self.trees:
            pred = tree.predict_proba(X)
            pred_sum += pred

        class_indecies = np.argmax(pred_sum, axis=1)
        preds = [self.classes[idx] for idx in class_indecies]
        return preds