from boosted_random_forest import *
from classic_random_forest import *
from utils import *
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

# set parameters
num_of_tests = 50
plot_cf = False
save_cf = False
crf_conf_matrix_save_path = None
brf_conf_matrix_save_path = None
num_of_trees = 100


files = ["data/tic-tac-toe.data",
        "data/wine.data",
        "data/obesity_levels.csv"]
file_path = files[0]

# program
encode_data, class_position, split_for_training = set_data_parameters(file_path=file_path)

X, y = load_data_from_file(file_path, class_position, encode_data=encode_data)
classes = np.unique(y)
n_classes = len(classes)

y_train = []
while len(np.unique(y_train)) != n_classes:
    X_train, X_test, y_train, y_test = data_train_test_split(X, y, split_for_training=split_for_training)


classic = 0
boosted = 0
draw = 0
crf_data = {"acc": [], "precision": [], "recall": [], "F1": []}
brf_data = {"acc": [], "precision": [], "recall": [], "F1": []}
for i in range(num_of_tests):
    print(f'Test nr. {i}')
    # classic RF
    crf = ClassicRandomForest(n_classes=n_classes, classes=classes, n_estimators=num_of_trees)
    crf.fit(X_train, y_train)
    y_pred_crf = crf.predict(X_test)
    crf_accuracy = accuracy_score(y_test, y_pred_crf)
    crf_precision = precision_score(y_test, y_pred_crf, average='macro')
    crf_recall = recall_score(y_test, y_pred_crf, average='macro')
    crf_f1 = f1_score(y_test, y_pred_crf, average='macro')
    if plot_cf:
        plot_conf_matrix(y_test, y_pred_crf, save=save_cf, save_path=crf_conf_matrix_save_path)
    # print(f'CRF acc: {crf_accuracy}, precision: {crf_precision}, recall: {crf_recall}, F1: {crf_f1}')

    # boosted RF
    brf = BoostedRandomForest(n_classes=n_classes, classes=classes, n_estimators=num_of_trees)
    brf.fit(X_train, y_train)
    y_pred_brf = brf.predict(X_test)
    brf_accuracy = accuracy_score(y_test, y_pred_brf)
    brf_precision = precision_score(y_test, y_pred_brf, average='macro')
    brf_recall = recall_score(y_test, y_pred_brf, average='macro')
    brf_f1 = f1_score(y_test, y_pred_brf, average='macro')
    if plot_cf:
        plot_conf_matrix(y_test, y_pred_brf, save=save_cf, save_path=brf_conf_matrix_save_path)
    # print(f'BRF acc: {brf_accuracy}, precision: {brf_precision}, recall: {brf_recall}, F1: {brf_f1}')
    # print()

    crf_data["acc"].append(crf_accuracy)
    crf_data["precision"].append(crf_precision)
    crf_data["recall"].append(crf_recall)
    crf_data["F1"].append(crf_f1)

    brf_data["acc"].append(brf_accuracy)
    brf_data["precision"].append(brf_precision)
    brf_data["recall"].append(brf_recall)
    brf_data["F1"].append(brf_f1)

    if crf_accuracy > brf_accuracy:
        classic += 1
    elif crf_accuracy < brf_accuracy:
        boosted += 1
    else:
        draw += 1


df = pd.DataFrame({
    'CRF_acc': crf_data["acc"],
    'CRF_precision': crf_data["precision"],
    'CRF_recall': crf_data["recall"],
    'CRF_F1': crf_data["F1"],
    'BRF_acc': brf_data["acc"],
    'BRF_precision': brf_data["precision"],
    'BRF_recall': brf_data["recall"],
    'BRF_F1': brf_data["F1"],
})

print(df)


print(f'classic better {classic} times ')
print(f'boosted better {boosted} times ')
print(f'draw {draw} times ')
