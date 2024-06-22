import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

class MyBoostedRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        self.alphas = []
        self.random_state = random_state

    def fit(self, X, y):
        N = len(y)
        M = len(np.unique(y))
        self.w = np.full(N, 1/N)
        self.classes = [cl for cl in np.unique(y)]

        # rng = np.random.default_rng(self.random_state)

        while len(self.trees) != self.n_estimators:
            # Step 4: Wybierz podzbiór Dt ze zbioru treningowego D
            # X_resampled, y_resampled, w_resampled = resample(X, y, w, n_samples=N, random_state=t)
            bootstrap_indices = np.random.choice(N, size=N, replace=True, p=self.w)
            # bootstrap_indices = rng.choice(N, size=N, replace=True, p=self.w)
            X_bs = []
            y_bs = []
            for idx in bootstrap_indices:
                X_bs.append(X[idx].copy())
                y_bs.append(y[idx].copy())
            X_bootstrap, y_bootstrap = np.array(X_bs), np.array(y_bs)

            # Step 5: Zbuduj drzewo decyzyjne Tt na podstawie Dt
            tree = DecisionTreeClassifier(max_features=self.max_features, criterion='entropy', random_state=self.random_state)
            tree.fit(X_bootstrap, y_bootstrap)

            # Step 16: Klasyfikuj użyte próbki Dt za pomocą drzewa
            y_pred = tree.predict(X)

            # Step 17: Oblicz błąd dla drzewa
            misclassified = (y_pred != y)
            w_sum = 0
            misclassified_indicies = bootstrap_indices[misclassified]
            unique_misclassified_w_indicies = np.unique(misclassified_indicies)
            for idx in unique_misclassified_w_indicies:
                w_sum += self.w[idx]
            epsilon_t = w_sum / np.sum(self.w)


            # Step 18: Oblicz wagę dla drzewa
            alpha_t = 0.5 * np.log((M - 1) * (1 - epsilon_t) / epsilon_t)
            # print(alpha_t)
            if alpha_t > 0:
                self.trees.append(tree)
                self.alphas.append(alpha_t)
                print(f'added tree nr. {len(self.trees)+1}')

                # Step 20: Aktualizuj wagi dla próbek w zbiorze treningowym
                # w = w * np.exp(alpha_t * misclassified * ((M - 1) / M))
                unique_misclassified_w_indicies = np.unique(misclassified_indicies)
                change_w = []
                for idx in range(len(self.w)):
                    if idx in unique_misclassified_w_indicies:
                        change_w.append(1)
                    else:
                        change_w.append(-1)
                change_w = np.array(change_w)
                self.w = self.w * np.exp(alpha_t * change_w)
                self.w /= np.sum(self.w)

    def predict(self, X):
        # Suma ważonych predykcji z poszczególnych drzew
        pred_sum = np.zeros((X.shape[0], len(self.trees[0].classes_)))

        for alpha, tree in zip(self.alphas, self.trees):
            pred = tree.predict_proba(X)
            pred_sum += alpha * pred

        class_indecies = np.argmax(pred_sum, axis=1)
        preds = [self.classes[idx] for idx in class_indecies]
        return preds