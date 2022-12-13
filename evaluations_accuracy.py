# coding=utf-8
"""
Accuracy evaluations.
"""

# %%
from sklearn.metrics import confusion_matrix
import numpy as np


# %%

def get_preds_list(node, preds):
    if node['is_terminal']:
        preds.append(node['pred'])
    else:
        get_preds_list(node['left'], preds)
        get_preds_list(node['right'], preds)


def is_constant(node):
    preds = []
    get_preds_list(node, preds)
    n_preds = len(set(preds))
    return n_preds == 1


def get_distance_to_split(x, pred_idx, w, euclidian=True, arr=None):
    if w is None:
        return arr

    if euclidian:
        x_i = x[:, pred_idx]
        ss = type(w)
        if type(w) == np.float64:
            cut = w
        else:
            cut = -w[0] / w[1]
        distances_boundary = x_i - cut
        if type(w) != np.float64 and w[1] < 0:
            distances_boundary = -1. * distances_boundary
    else:
        x_i = x[:, [0, pred_idx]]
        distances_boundary = np.dot(w, x_i.T)

    return np.array(distances_boundary, dtype=np.float64)


def get_split_predictions(x, pred_idx, w):
    """
    ----------
    Parameters

    param w: ndarray
    param x_test: {array-like, sparse matrix}, shape (n_test_samples, n_features). Test data.
    return: ndarray, shape (n_test_samples,). Array of predictions.
    -----
    Notes

    Either pass cut or pass y_test_predicted.

    """
    distances_boundary_test = get_distance_to_split(x, pred_idx, w)
    y_test_predicted = np.sign(distances_boundary_test)
    return y_test_predicted


def get_split_accuracy(pred_idx, w, x_test, y_test, y_test_predicted=None):
    """
    ----------
    Parameters

    param w: ndarray
    param x_test: {array-like, sparse matrix}, shape (n_test_samples, n_features+1). Test data.
    param y_test: ndarray, shape (n_test_samples,). Array of test labels.
    param y_test_predicted: ndarray, shape (n_test_samples,). Array of predictions.
    return: the test accuracy of the model
    -----
    Notes

    Either pass cut or pass y_test_predicted.

    """

    if w is not None and y_test_predicted is not None:
        raise Exception("Either the cut (w) or the predicted labels should be None")
    if w is not None:
        y_test_predicted = get_split_predictions(x_test, pred_idx, w)
    correct_answers = y_test_predicted == y_test
    test_score = float(np.sum(correct_answers)) / float(correct_answers.shape[0])
    return test_score


def get_accuracy(model, dataset_ext, test=True, val=False):
    """
    Calculate accuracy percentage

    -----------max
    Parameters

    :param model:
    :param dataset_ext:
    return: float. Accuracy percentage.

    """
    assert not test or not val
    if test:
        actual = dataset_ext.y_test
        x = dataset_ext.x_test
    elif val:
        actual = dataset_ext.y_val
        x = dataset_ext.x_val
    else:
        actual = dataset_ext.y_train
        x = dataset_ext.x_train

    predicted = prediction_set(model, x)
    correct_answers = actual == predicted
    return np.sum(correct_answers) / float(len(correct_answers)) * 100.0


def get_tree_predictions_and_trues(model, dataset_ext, test=True, val=False, is_tree=True):
    """
    Calculate accuracy percentage

    -----------max
    Parameters

    :param model:
    :param dataset_ext:
    return: float. Accuracy percentage.

    """
    assert not test or not val
    if test:
        actual = dataset_ext.y_test
        x = dataset_ext.x_test
    elif val:
        actual = dataset_ext.y_val
        x = dataset_ext.x_val
    else:
        actual = dataset_ext.y_train
        x = dataset_ext.x_train

    predicted = prediction_set(model, x, is_tree)
    return predicted, actual


def predict(node, row):
    """
    Make a prediction with a decision tree

    ----------
    Parameters

    param node: dict. A nested dict representing a binary tree.
    For first call use node = the decision tree.
    param row: ndarray, shape (n_predictors,). Single case to apply prediction.
    return: int. Classification of row made by tree.

    """
    if node['is_terminal']:
        return node['pred']

    if row[node['index']] < node['cut']:
        if not node['left']['is_terminal']:
            return predict(node['left'], row)
        else:
            return node['left']['pred']
    else:
        if not node['right']['is_terminal']:
            return predict(node['right'], row)
        else:
            return node['right']['pred']


def prediction_set(model, test, is_tree=True):
    """
    Parameters

    param tree: dict. Decision tree.
    param test: ndarray, shape (n_test, n_predictors). Test set.
    return: ndarray, shape (n_test, ). Predictions made by tree.

    """

    if not is_tree:
        predictions = np.sign(np.dot(test, model))
        return predictions

    predictions = np.zeros(test.shape[0])
    for i, case in enumerate(test):
        predictions[i] = predict(model, case)
    return predictions


def accuracy_metric(actual, predicted):
    """
    Calculate accuracy percentage

    Parameters

    param actual: ndarray, shape (n_test,). True labels y.
    param predicted: ndarray, shape (n_test,). Predicted labels of a model.
    return: float. Accuracy percentage.

    """
    correct_answers = actual == predicted
    return np.sum(correct_answers) / float(correct_answers.shape[0]) * 100.0


def get_group_pred(group, test=False):
    if test:
        y = group.y_test
    else:
        y = group.y_train
    outcomes = list(y)
    return max(set(outcomes), key=outcomes.count)


def get_group_pred_and_acc(group, test=False):
    pred = get_group_pred(group, test=test)
    if test:
        y = group.y_test
    else:
        y = group.y_train
    pred_arr = np.zeros_like(y) + pred
    correct_answers = pred_arr == y
    acc = float(np.sum(correct_answers)) / float(correct_answers.shape[0])
    return pred, acc


#%%
def get_confusion_matrices(y_true, y_pred, protected_vals):

    y_true_prot = y_true[protected_vals == 0]
    y_true_non_prot = y_true[protected_vals == 1]
    y_pred_prot = y_pred[protected_vals == 0]
    y_pred_non_prot = y_pred[protected_vals == 1]

    labels = np.array([-1, 1])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_prot = confusion_matrix(y_true_prot, y_pred_prot, labels=labels)
    cm_non_prot = confusion_matrix(y_true_non_prot, y_pred_non_prot, labels=labels)

    return cm, cm_prot, cm_non_prot


def get_true_and_neg_rates(conf_matrix):
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]

    try:
        tpr = float(tp) / float(tp + fn)
    except ZeroDivisionError:
        tpr = np.nan
    try:
        tnr = float(tn) / float(tn + fp)
    except ZeroDivisionError:
        tnr = np.nan

    return tpr, tnr


def get_balanced_accuracy(conf_matrix):
    tpr, tnr = get_true_and_neg_rates(conf_matrix)
    return 0.5 * (tpr + tnr)


def get_group_acc(group, pred, test=False):
    if test:
        y = group.y_test
    else:
        y = group.y_train

    pred_arr = np.zeros_like(y) + pred
    correct_answers = pred_arr == y
    acc = float(np.sum(correct_answers)) / float(correct_answers.shape[0])
    return acc

