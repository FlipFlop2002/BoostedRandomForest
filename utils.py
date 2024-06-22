import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def encode_data(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()

    # Podział danych na atrybuty i etykiety
    X = []
    y = []
    for line in lines:
        parts = line.strip().split(',')
        X.append(parts[:-1])  # atrybuty
        y.append(parts[-1])   # etykieta

    # Konwersja danych na numeryczne
    label_encoder = LabelEncoder()
    X_encoded = []
    for i in range(len(X[0])):
        feature_values = [row[i] for row in X]
        if all(is_numeric(value) for value in feature_values):
            X_encoded.append([float(value) for value in feature_values])
        else:
            feature_encoded = label_encoder.fit_transform(feature_values)
            X_encoded.append(feature_encoded)
    X_encoded = np.array(X_encoded).T
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded

def encode_data_old(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()

    # Podział danych na atrybuty i etykiety
    X = []
    y = []
    for line in lines:
        parts = line.strip().split(',')
        X.append(parts[:-1])  # atrybuty
        y.append(parts[-1])   # etykieta

    # Konwersja danych na numeryczne
    label_encoder = LabelEncoder()
    X_encoded = []
    for i in range(len(X[0])):
        feature_values = [row[i] for row in X]
        feature_encoded = label_encoder.fit_transform(feature_values)
        X_encoded.append(feature_encoded)
    X_encoded = np.array(X_encoded).T
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded


def load_data_from_file(file_path, class_position, encode_data: bool):
    """
    Ładuje dane z pliku .data, uwzględniając pozycję klasy.

    :param file_path: Ścieżka do pliku .data
    :param class_position: Pozycja klasy ('first' lub 'last')
    :return: Cechy (X) i etykiety (y)
    """
    X = []
    y = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if class_position == 'first':
                y.append(row[0])
                X.append(row[1:])
            elif class_position == 'last':
                y.append(row[-1])
                X.append(row[:-1])
            else:
                raise ValueError("class_position must be 'first' or 'last'")

    if encode_data:
        # Konwersja danych na numeryczne
        label_encoder = LabelEncoder()
        X_encoded = []
        for i in range(len(X[0])):
            feature_values = [row[i] for row in X]
            if all(is_numeric(value) for value in feature_values):
                # Jeśli wszystkie wartości są liczbowe, konwertuj bezpośrednio na float
                X_encoded.append([float(value) for value in feature_values])
            else:
                # W przeciwnym razie użyj LabelEncoder
                feature_encoded = label_encoder.fit_transform(feature_values)
                X_encoded.append(feature_encoded)

        # Transponowanie tablicy dla poprawnego kształtu
        X_encoded = np.array(X_encoded).T
        # Konwersja etykiet klas na wartości liczbowe
        y_encoded = label_encoder.fit_transform(y)
        return X_encoded, y_encoded

    return np.array(X), np.array(y)

def data_train_test_split(X, y, split_for_training=True):
    if split_for_training:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test
    else:
        return X, X, y, y

def set_data_parameters(file_path: str):
    if file_path == "data/tic-tac-toe.data":
        encode_data = True
        class_position = "last"
        split_for_training = True
    elif file_path == "data/wine.data":
        encode_data = False
        class_position = "first"
        split_for_training = True
    elif file_path == "data/obesity_levels.csv":
        encode_data = True
        class_position = "last"
        split_for_training = True
    else:
        raise ValueError("Wrong data path")

    return encode_data, class_position, split_for_training

def plot_conf_matrix(y_test, y_pred, save: bool, save_path=None):
    # macierz pomyłek
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    plt.title('Macierz Pomyłek')
    plt.colorbar()

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="black")

    plt.xlabel('przewidywana etykieta')
    plt.ylabel('rzeczywista etykieta')
    plt.xticks(np.unique(y_test))
    plt.yticks(np.unique(y_test))
    if save:
        plt.savefig(save_path)
    plt.show()