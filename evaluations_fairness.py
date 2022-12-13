# coding=utf-8
"""
Accuracy and fairness evaluations of lineal boundary models.
"""

# %%
from collections import defaultdict
from sklearn.metrics import accuracy_score
from copy import deepcopy
import pandas as pd

from tree_optimization_fair_constrains import get_all_covariance_sensitive_attrs
from evaluations_accuracy import *
from data_utils import check_two_sensible_classes


# %%

def get_depth(node):
    if not node['is_terminal']:
        max_depht_left = get_depth(node['left'])
        max_depht_right = get_depth(node['right'])
        return max(max_depht_left, max_depht_right)
    else:
        return node['depth']


def node_to_terminal(node):
    """
    Setting a branch to terminal, auxiliary for constructing and prunning.
    -----
    Parameters

    param node: dict. Node of a decision tree.
    return: dict. Node with format of terminal.
    """
    node['n_terminals'] = 1
    node['is_terminal'] = True

    if 'left' in node.keys():
        del (node['left'])
        del (node['right'])

    return


def prune_by_depth(node, depth):
    # prune by depth
    if not node['is_terminal'] and node['depth'] >= depth:
        node_to_terminal(node)

    # recursion if not terminal
    if not node['is_terminal']:
        node['left'] = prune_by_depth(node['left'], depth)
        node['right'] = prune_by_depth(node['right'], depth)

    return node


def count_variable_types(node, depth, continous_idx, n_training):
    assert (node['depth'] <= depth)

    if node['is_terminal']:
        return np.zeros(4)

    if node['depth'] < depth:
        counts_l = count_variable_types(node['left'], depth, continous_idx, n_training)
        counts_r = count_variable_types(node['right'], depth, continous_idx, n_training)
        return counts_l + counts_r

    elif node['depth'] == depth:
        n_discrete, n_continuous, n_cases_dis, n_cases_con = np.zeros(4)
        if 'index' in node.keys() and node['index'] in continous_idx:
            n_continuous = 1
            n_cases_con = node['group'].training_size * 100. / n_training
        else:
            n_discrete = 1
            n_cases_dis = node['group'].training_size * 100. / n_training
        return np.round(np.array([n_discrete, n_continuous, n_cases_dis, n_cases_con]), 2)


def get_metrics_of_depth(tree, dataset_ext, depth, test=True, val=False):
    max_depth = get_depth(tree)
    if max_depth < depth:
        return -1, -1, -1

    tree_prunned = deepcopy(tree)
    tree_prunned = prune_by_depth(tree_prunned, depth)
    assert (get_depth(tree_prunned) == depth)

    accuracy = get_accuracy(tree_prunned, dataset_ext, test=test, val=val)
    correlation_dict = get_model_correlations(tree_prunned, dataset_ext, test=test, val=val)
    s_attr_name = dataset_ext.sensitive_attrs[0]
    p_value, n_value = p_rules_model_stats(correlation_dict, s_attr_name)

    return accuracy, p_value, n_value


def get_covs_of_depth(node, depth, covs):
    if not node['is_terminal']:
        if node['depth'] == depth:
            covs.append(node['mean_cov'])
        elif node['depth'] > depth:
            return
        else:
            get_covs_of_depth(node['left'], depth, covs)
            get_covs_of_depth(node['right'], depth, covs)


def get_correlations(model, x, predictions, sensitive, sensitive_attrs, percentage=True):
    """
    This function is used from arXiv:1507.05259 [stat.ML] source code below.

    ----------
    Parameters

    param model: ndarray, shape (n_features,) or (n_features + 1,). Coefficient vector determinig the model.
    param x: {array-like, sparse matrix}, shape (n_test_samples, n_features). Test data.
    param predictions: ndarray, shape (n_test_samples,). Array of predictions.
    param sensitive: dictionary: key= sensitive attribute name, value = ndarray, shape (n_samples,),
    an array of sensitive features.
    param sensitive_attrs: list of strings. Names of sensitive attributtes.
    param pred: if the predictions of the model are all the same (e.g. model is the terminal of a tree)
    return: nested dictionary:
    - Outer dictionary (associated with sensitive attributes): key = name of sensitive attribute, and value= middle
     dictionary
    - Middle dictionary (associated with the categories of the attribute): key = category_of_attribute, and
    value = inner dictionary
    - Inner dictionary (associated with the type of predictions in category): key = class of predictions (1 or -1),
    value = percentage of the total persons in the category, assignated to that prediction

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    if model is not None:
        predictions = np.sign(np.dot(x, model))

    predictions = np.array(predictions)

    out_dict = {}
    if sensitive_attrs:
        for attr in sensitive_attrs:
            attr_val = sensitive[attr]
            assert (len(attr_val) == len(predictions))

            total_per_val = defaultdict(int)  # diccionario que no da errores al buscar llaves no presentes, regresa 0
            # TODO: quizas habria que cambiarlo para que regrese 1, hay division entre ese numero
            attr_to_class_labels_dict = defaultdict(lambda: defaultdict(int))

            for i in range(0, len(predictions)):
                val = attr_val[i]
                label = predictions[i]
                # val = attr_val_int_mapping_dict_reversed[val] # change values from intgers to actual names
                total_per_val[val] += 1  # conteo de personas con categora val del atributo attr
                attr_to_class_labels_dict[val][label] += 1

            class_labels = set(predictions.tolist())
            local_dict_1 = defaultdict(lambda: defaultdict(int))
            for k1, v1 in attr_to_class_labels_dict.items():
                total_this_val = total_per_val[k1]

                local_dict_2 = defaultdict(lambda: 0.)
                for k2 in class_labels:
                    v2 = v1[k2]
                    f = float(v2) * 100.0 / float(total_this_val)
                    local_dict_2[k2] = f

                local_dict_1[k1] = local_dict_2
            out_dict[attr] = local_dict_1

    return out_dict


# get_node_fair_statistics(group, pred_idx_best, cut_best, group.sensitive_attrs, test=False)
def get_node_fair_statistics(group, pred_idx, w, sensitive_attrs, test=False):
    """
    Fairness report

    ---------
    Parameters

    :param pred_idx: the predictor that will be splited
    :return:
            p-rule: percentage between subjects having a certain sensitive attribute value
            and the subjects not having that value, both groups assigned the positive decision outcome
    """
    if test:
        sensitive = group.sensitive_test
        x = group.x_test
    else:
        sensitive = group.sensitive_train
        x = group.x_train

    two_sensible_classes = check_two_sensible_classes(sensitive, sensitive_attrs)

    # TODO: extender la excepción para más de un atributo sensible:
    if not two_sensible_classes[sensitive_attrs[0]]:  # only one sensible class in the test
        p, n, cov = 0, 0, 0
    else:

        # data for evaluations
        distances_boundary = get_distance_to_split(x, pred_idx, w)
        predicted = np.sign(distances_boundary)

        # mean cov
        x_i = x[:, pred_idx]
        covs = get_all_covariance_sensitive_attrs(None, x_i, distances_boundary, sensitive,
                                                  group.sensitive_attrs, get_dict=False)
        cov = np.mean(covs)

        # p and n values
        correlation_dict = get_correlations(None, None, predicted, sensitive,
                                            group.sensitive_attrs)
        p, n = p_rules_stats([correlation_dict], group.sensitive_attrs[0])

    return p, n, cov


def get_terminal_fair_statistics(group, pred, test=False):
    if test:
        sensitive_values = group.sensitive_test.values()
        x = group.x_test
        sensitive = group.sensitive_test
    else:
        sensitive_values = group.sensitive_train.values()
        x = group.x_train
        sensitive = group.sensitive_train

    classes_in_sensitive_attrs = [np.unique(arr).shape[0] for arr in sensitive_values]
    if 1 in classes_in_sensitive_attrs:
        one_sensible_class = True
    else:
        one_sensible_class = False

    if one_sensible_class:
        p, n = 0.0, 0.0
    else:
        predicted = np.array([pred] * x.shape[0])
        correlation_dict = get_correlations(model=None, x=None, predictions=predicted,
                                            sensitive=sensitive,
                                            sensitive_attrs=group.sensitive_attrs)

        p, n = p_rules_stats([correlation_dict], group.sensitive_attrs[0])

    return p, n


def get_attr_pred_to_count_dict(preds, sensitive_vals):
    """

    ----------
    Parameters

    param preds: ndarray, shape (n_test_samples,). Array of predictions.
    param x_sensitive_test: value = ndarray, shape (n_samples,),
    an array of sensitive features associated to same data of preds.
    param sensitive_attrs: list of strings. Names of sensitive attributtes.
    param pred: if the predictions of the model are all the same (e.g. model is the terminal of a tree)
    return: nested dictionary:
    - outer dictionary: key = category_of_sensitive_attribute, and
    value = inner dictionary
    - Inner dictionary (associated with the type of predictions): key = class of predictions (1 or -1),
    value = total persons in the category_of_sensitive_attribute, assignated to that prediction

    -----

    """

    assert (sensitive_vals.shape[0] == preds.shape[0])

    attr_pred_to_count_dict = defaultdict(lambda: defaultdict(int))

    for i in range(preds.shape[0]):
        sensitive_cat = sensitive_vals[i]
        prediction_name = preds[i]
        attr_pred_to_count_dict[sensitive_cat][prediction_name] += 1  # count of sensitive_cat of
        # attribute attr with assigned to prediction

    return attr_pred_to_count_dict


def get_model_correlations(model, dataset_ext, test=True, val=False, percentage=True, is_tree=True):
    assert not test or not val
    if test:
        x = dataset_ext.x_test
        sensitive = dataset_ext.sensitive_test
    elif val:
        x = dataset_ext.x_val
        sensitive = dataset_ext.sensitive_val
    else:
        x = dataset_ext.x_train
        sensitive = dataset_ext.sensitive_train

    predicted = prediction_set(model, x, is_tree=is_tree)

    if percentage:
        correlation_dict = get_correlations(None, None, predicted, sensitive,
                                             dataset_ext.sensitive_attrs)
    else:
        sensitive_vals = sensitive[dataset_ext.sensitive_attrs[0]]
        correlation_dict = get_attr_pred_to_count_dict(predicted, sensitive_vals)

    return correlation_dict


def statistical_parity_difference(dataset, protected, classes, majority_group, minority_group, value):
    ratio = round(pd.crosstab(dataset[protected], dataset[classes]).div(pd.crosstab(dataset[protected], dataset[classes]).apply(sum,1),0),4)*100
    return ratio


def get_model_one_p_value(correlation_dict, s_attr_name):
    """
    Fairness report

    ---------
    Parameters

    :param model:
    :param dataset_ext:
    :return:
    - p-rule: percentage between subjects having 1 sensitive attribute value
            and the subjects not having that value, both groups assigned the positive decision outcome
    """

    non_prot_pos = correlation_dict[s_attr_name][1][1]
    prot_pos = correlation_dict[s_attr_name][0][1]

    if non_prot_pos > 0:
        p_rule = (prot_pos / non_prot_pos)
    elif prot_pos > 0:
        p_rule = np.inf
    else:  # al cases are in negative class so p_rule = 0/0
        p_rule = 1.
    return p_rule


def p_rules_stats(correlation_dict_arr, s_attr_name, epsilon=10e-5):
    """
    This function is adapted from arXiv:1507.05259 [stat.ML] source code below.

    ----------
    Parameters

    param correlation_dict_arr: array of dicts, len = number of sensitive attributes.
    Each dict of the returned type of get_correlations.
    param s_attr_name: string. One sensitive attribute name.
    return:
     - p_rule: float. Value of the p-rule
     - n_rule: float. Value of the n-rule cocient.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    # TODO: pensar por que quieren tener varios diccionarios (el argumento es un arreglo)
    # TODO: pensar para que usaban esa función: correlation_dict = get_avg_correlation_dict(correlation_dict_arr)
    correlation_dict = correlation_dict_arr[0]
    non_prot_pos = correlation_dict[s_attr_name][1][1]
    prot_pos = correlation_dict[s_attr_name][0][1]

    if prot_pos > epsilon and non_prot_pos > epsilon:
        p_rule = min(prot_pos / non_prot_pos * 1., non_prot_pos / prot_pos * 1.)
    elif non_prot_pos < epsilon and prot_pos < epsilon:
        # all are classified w pred=0, p_rule = 0/0
        p_rule = 1.
    elif non_prot_pos < epsilon:
        # pred 1 not in non-protected group
        p_rule = 0.
    elif prot_pos < epsilon:
        # pred 1 not in protected group
        p_rule = 0.

    non_prot_neg = correlation_dict[s_attr_name][1][-1]
    prot_neg = correlation_dict[s_attr_name][0][-1]

    if prot_neg > epsilon and non_prot_neg > epsilon:
        n_rule = min(prot_neg * 1. / non_prot_neg, non_prot_neg * 1. / prot_neg)
    elif non_prot_neg < epsilon and prot_neg < epsilon:
        # all are classified w pred=1, n_rule = 0/0
        n_rule = 1.
    elif non_prot_neg < epsilon:
        # pred 0 not in non-protected group
        n_rule = 0.
    elif prot_neg < epsilon:
        # pred 0 not in protected group
        n_rule = 0.

    return p_rule, n_rule



def p_rules_model_stats(correlation_dict=None, s_attr_name=None, epsilon=10e-5, non_prot_pos=None, prot_pos=None,
                        non_prot_neg=None, prot_neg=None, multiply=1.0):
    """
    This function is adapted from arXiv:1507.05259 [stat.ML] source code below.

    ----------
    Parameters

    param correlation_dict_arr: array of dicts, len = number of sensitive attributes.
    Each dict of the returned type of get_correlations.
    param s_attr_name: string. One sensitive attribute name.
    return:
     - p_rule: float. Value of the p-rule
     - n_rule: float. Value of the n-rule cocient.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    if non_prot_pos == None:
        non_prot_pos = correlation_dict[s_attr_name][1][1]
        prot_pos = correlation_dict[s_attr_name][0][1]

    if prot_pos > epsilon and non_prot_pos > epsilon:
        p_rule = min(prot_pos / non_prot_pos * 1., non_prot_pos / prot_pos * 1.) * multiply
    elif prot_pos < epsilon < non_prot_pos:
        p_rule = (prot_pos / non_prot_pos * 1.) * multiply
    elif prot_pos > epsilon > non_prot_pos:
        p_rule = (non_prot_pos / prot_pos * 1.) * multiply
    else:
        p_rule = 1. * multiply

    if non_prot_neg == None:
        non_prot_neg = correlation_dict[s_attr_name][1][-1]
        prot_neg = correlation_dict[s_attr_name][0][-1]

    if prot_neg > epsilon and non_prot_neg > epsilon:
        n_rule = min(prot_neg * 1. / non_prot_neg, non_prot_neg * 1. / prot_neg) * multiply
    elif prot_neg < epsilon < non_prot_neg:
        n_rule = (prot_neg / non_prot_neg * 1.) * multiply
    elif prot_neg > epsilon > non_prot_neg:
        n_rule = (non_prot_neg / prot_neg * 1.) * multiply
    else:
        n_rule = 1. * multiply

    return p_rule, n_rule


def calculate_performance_sp(trues, predictions, protected_vals):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    for pred, prot_attr in zip(predictions, protected_vals):
        if prot_attr == 0:
            if pred == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.
        else:
            if pred == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

    C_prot = protected_pos / (protected_pos + protected_neg)
    C_non_prot = non_protected_pos / (non_protected_pos + non_protected_neg)
    stat_par = C_non_prot - C_prot
    accuracy = accuracy_score(trues, predictions)

    return stat_par, accuracy


def calculate_performance_fair_mine(y_pred, protected_vals, epsilon=10e-8):

    y_pred_prot = y_pred[protected_vals == 0]
    y_pred_non_prot = y_pred[protected_vals == 1]

    protected_pos = np.sum(y_pred_prot == 1)
    protected_neg = np.sum(y_pred_prot == -1)
    non_protected_pos = np.sum(y_pred_non_prot == 1)
    non_protected_neg = np.sum(y_pred_non_prot == -1)

    # Statistical parity
    C_prot = protected_pos / (protected_pos + protected_neg)
    C_non_prot = non_protected_pos / (non_protected_pos + non_protected_neg)
    stat_par = C_non_prot - C_prot

    # p-rule
    if protected_pos > epsilon and non_protected_pos > epsilon:
        p_rule = min(protected_pos / non_protected_pos * 100., non_protected_pos / protected_pos * 100.)
    elif protected_pos < epsilon < non_protected_pos:
        p_rule = (protected_pos / non_protected_pos * 100.)
    elif protected_pos > epsilon > non_protected_pos:
        p_rule = (non_protected_pos / protected_pos * 100.)
    else:
        p_rule = 100.


    if protected_neg > epsilon and non_protected_neg > epsilon:
        n_rule = min(protected_neg * 100. / non_protected_neg, non_protected_neg * 100. / protected_neg)
    elif protected_neg < epsilon < non_protected_neg:
        n_rule = (protected_neg / non_protected_neg * 100.)
    elif protected_neg > epsilon > non_protected_neg:
        n_rule = (non_protected_neg / protected_neg * 100.)
    else:
        n_rule = 100.

    return stat_par, p_rule, n_rule


