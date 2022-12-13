# coding = utf-8
# %%
import numpy as np

from tree_construction import *


# %%

def get_gini(group):
    """
    Calculate the Gini index for a dataset.
    gini(t) = 1-sum_{j=1}^Jp_j^2
    :param group:
    :return:
    """

    size = float(group.training_size)
    if size == 0.:
        return 1.

    sum = 0.
    for class_val in group.classes_names:
        class_vals = np.zeros_like(group.y_train) + class_val
        p = float(np.sum(group.y_train == class_vals))/size
        sum += p * p

    return 1. - sum


def get_gini_cut(pred_index, group):
    """
    Calculate best gini cut, given a predictor.
    
    param pred_index: int the predictor.
    param group: DatasetExt instance. Cases present in the node of interest.
    :return: 
    """

    size_node = float(group.training_size)
    cut_best, gini_best, groups_best, p_best = None, np.inf, None, None
    for row in group.x_train:
        left, right = split_cases_by_cut(pred_index, row[pred_index], group)
        p_L, p_R = float(left.training_size)/size_node, float(right.training_size)/size_node
        gini = p_L * get_gini(left) + p_R * get_gini(right)
        if gini < gini_best:
            cut_best, gini_best, groups_best = row[pred_index], gini, (left, right)

    return cut_best, gini_best, groups_best


def get_gini_split(group, depth):
    """
    Select the best split point for a dataset
    param group: DatasetExt instance.
    :return: 
    """
    pred_idx_best, cut_best, gini_best, groups_best, R_best = None, None, np.inf, None, None

    for pred_idx in group.predictors_idx:
        cut, gini, groups = get_gini_cut(pred_idx, group)
        if gini < gini_best:
            pred_idx_best, cut_best, gini_best, groups_best = pred_idx, cut, gini, groups

    pred = None
    acc = None
    R_fair = None

    if groups_best is not None:
        R_best, pred = estimate_mis_node(group)
        acc = get_group_acc(group, pred, test=False)
        R_fair = 1. - estimate_rule_node(group)
        assert 0. <= R_fair <= 1.

    p, n, cov = get_node_fair_statistics(group, pred_idx_best, cut_best, group.sensitive_attrs, test=False)

    return {'index': pred_idx_best, 'cut': cut_best, 'group': group, 'groups': groups_best,
            'R_node': R_best, 'R_branch': R_best, 'R_fair': R_fair, 'p_value': p, 'n_value': n,
            'mean_cov': cov, 'pred': pred, 'acc': acc, 'depth': depth}

