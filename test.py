import numpy as np


# miss = np.array([True, True, False, True])
# w = np.array([0.1, 0.3, 0.5, 0.1])
# print(w[miss])

indicies = np.array([0, 5, 8, 2, 5, 5, 6, 0])
print(np.unique(indicies))
print(8 in indicies)


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues')
plt.show()