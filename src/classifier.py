import os
import collections
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import globals

MODEL_DIR = globals.MODEL_DIR
PLOT_DIR = globals.PLOT_DIR

plt.rc('text', usetex=globals.usetex)
plt.rc('font', family='serif')

C_1_FNAME = 'c_1_svm'
C_BEST_FNAME = 'c_best_svm'

C_VALS = [0.001, 0.01, 0.1, 1, 10, 100] #, 1000]


def fit_clf(X, Y, C=1.0, weight=None, linearsvc=True, fname=None):
    """
    Trains a linear SVM model, either using a given C or by performing a grid
    search over a range of C values and/or class weights.
    :param X: Training set features
    :param Y: Training set labels
    :param C: The value(s) of hyperparameter C
    :param weight: The weight of classes
    :param liblinear: Whether to use LinearSVC (LibLinear) or svm.SVC (LibSVM)
    :param fname: The name of the file to/from which the model is saved/loaded.

    :return: Trained SVM model
    """
    if np.isscalar(C):
        # Use the given C to train the classifier (no tuning of hyperparameter C)
        return __fit_c1_clf(X, Y, C, weight, linearsvc, fname)

    elif isinstance(C, (collections.Sequence, np.ndarray)):
        # Tune hyperparameter C over the given range of values
        return __fit_c_best_clf(X, Y, C, weight, linearsvc, fname)
    else:
        raise ValueError("Argument C must be a scalar, a Sequence or an ndarray")


def __fit_c1_clf(X, Y, C=1.0, weight=None, linearsvc=True, fname=None):
    """
    Trains an SVM classifier using a given value of C.
    :return: Trained SVM model
    """
    if fname is None:
        fname = C_1_FNAME
    model_path = MODEL_DIR + fname + '.joblib'

    if os.path.isfile(model_path):
        clf = joblib.load(model_path)
        print(f"Model {fname} Loaded")
    else:
        print(f"Training {fname}")
        if linearsvc:
            clf = LinearSVC(C=C, penalty='l2', loss='hinge', max_iter=1e6,
                            class_weight=weight, verbose=1)
        else:
            clf = SVC(C=C, kernel='linear', probability=True, max_iter=-1,
                  class_weight=weight, verbose=True)

        clf.fit(X, Y)
        print(f"Finished Training Model {fname}")
        joblib.dump(clf, model_path)
        print(f"Trained model saved to {model_path}")

    return clf


def __fit_c_best_clf(X, Y, C=C_VALS, weight=None, linearsvc=True, fname=None):
    """
    Performs a grid search with cross-validation to tune the hyperparameter C
    and/or class weights.
    :return: The model trained with the best parameters.
    """
    if fname is None:
        fname = C_BEST_FNAME
    model_path = MODEL_DIR + fname + '.joblib'

    if os.path.isfile(model_path):
        clf = joblib.load(model_path)
        print(f"Model {fname} Loaded")
    else:
        print("Tuning Hyperparameter(s) with Grid Search & Cross Validation")
        params = {'C': C, 'class_weight': weight}
        if linearsvc:
            clf = LinearSVC(penalty='l2', loss='hinge', max_iter=1e6, verbose=1)
        else:
            clf = SVC(C=C, kernel='linear', max_iter=-1, #probability=True,
                      class_weight=weight, verbose=True)

        grid_clf = GridSearchCV(clf, params, cv=5, scoring='f1',
                                n_jobs=-1, verbose=10).fit(X, Y)

        best_clf = grid_clf.best_estimator_
        print("Finished Grid Search. Best model:", best_clf)
        joblib.dump(best_clf, model_path)
        print(f"Best model saved to {model_path}")

    return best_clf


def evaluate(Y_true, Y_pred, subset='', cmap=plt.cm.Blues, save_fig=False):
    """
    Computes and outputs a number of evaluation metrics based on the given
    predicted and true labels.
    :param Y_true: The true class labels.
    :param Y_pred: The predicted class labels.
    :param subset: The name of the evaluated subset (eg, test), just for output.
    :param cmap: The colormap to use when plotting the confusion matrix.
    :param save_fig: Whether to save the generated figure.
    :return: None
    """
    # Text output
    accuracy = accuracy_score(Y_true, Y_pred)
    print(f'Accuracy: {accuracy:.4f}\n')

    report = classification_report(Y_true, Y_pred,
                                   labels=[1, 0],
                                   target_names=['Malware', 'Goodware'])
    print("Classification Report:\n", report)

    keys = np.array(['TN', 'FP', 'FN', 'TP'])
    print("Confusion matrix without Normalisation:")
    values = confusion_matrix(Y_true, Y_pred).ravel()
    metrics = list(zip(keys[::-1], values[::-1]))
    print(f'{metrics}\n')
    print("Confusion matrix with Normalisation:")
    values = np.round(confusion_matrix(Y_true, Y_pred, normalize='true').ravel(), 4)
    metrics = list(zip(keys[::-1], values[::-1]))
    print(f'{metrics}\n')

    # Confusion matrix plot
    f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(10, 5))
    f.suptitle("Confusion Matrix", fontsize=18)
    f.text(0.4, 0.05, 'Predicted label', ha='left', fontsize=16)
    f.text(0.05, 0.5, 'True label', ha='left', va='center', fontsize=16,
           rotation='vertical')
    labels = ['Malware', 'Goodware']
    plots = [(1, "Without Normalisation", None, 'd', ax1, True),
             (2, "With Normalisation", 'true', '.2f', ax2, False)]

    for i, title, normalize, fmt, ax, ylabel in plots:
        cm = confusion_matrix(Y_true, Y_pred,
                              labels=[1, 0],
                              normalize=normalize)
        sns.heatmap(cm, ax=ax, annot=True, square=True, fmt=fmt, cmap=cmap)
        ax.set_title(title)
        ax.set(xticklabels=labels,
               yticklabels=labels,
               xlabel='',
               ylabel='')

    if save_fig:
        plt.savefig(PLOT_DIR + f'confusion_matrix_{subset}.png',
                    dpi=1200, transparent=True)
    plt.show()

