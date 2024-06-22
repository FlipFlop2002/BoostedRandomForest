import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

class BoostedRandomForest:
    def __init__(self, n_classes, classes: list, n_estimators=100, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        self.alphas = []
        self.n_classes = n_classes
        self.classes = classes

    def fit(self, X, y):
        N = len(y)
        M = self.n_classes
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

            y_pred = tree.predict(X)

            misclassified = (y_pred != y)
            epsilon_t = np.sum(self.w[misclassified]) / np.sum(self.w)


            alpha_t = 0.5 * np.log((M - 1) * (1 - epsilon_t) / epsilon_t)

            if alpha_t > 0:
                self.trees.append(tree)
                self.alphas.append(alpha_t)

                change_w = []
                for i in misclassified:
                    if i:
                        change_w.append(1)
                    else:
                        change_w.append(-1)
                change_w = np.array(change_w)
                self.w = self.w * np.exp(alpha_t * change_w)
                self.w /= np.sum(self.w)

    def predict(self, X):
        # Suma ważonych predykcji z poszczególnych drzew
        pred_sum = np.zeros((X.shape[0], self.n_classes))

        for alpha, tree in zip(self.alphas, self.trees):
            tree_pred = tree.predict_proba(X)
            pred = np.zeros((X.shape[0], len(self.classes)))
            for idx in range(len(self.classes)):
                if self.classes[idx] in tree.classes_:
                    pred_idx = np.where(tree.classes_ == self.classes[idx])[0]
                    pred[:, idx] = tree_pred[:, pred_idx[0]]
            pred_sum += alpha * pred

        class_indecies = np.argmax(pred_sum, axis=1)
        preds = [self.classes[idx] for idx in class_indecies]
        return preds
