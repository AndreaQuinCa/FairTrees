# coding=utf-8
# %%
import numpy as np
from tree_prunning import *
from data_utils import *
from visualization_trees import *
from random import seed
from tree_construction_gini import get_gini_split

import pickle


# %%
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds, m_seed=110979):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            seed(m_seed)
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def get_dataset_folds(dataset_ext, n_folds):
    """
    Divides a dataset in n sub-datasets for cross validation (cv).

    --------
    Parameters

    :param dataset_ext: DatasetExt instance. The data to be partitioned
        should be in dataset_ext.data_tr and dataset_ext.data_ts.
    :param n_folds: int. Number of cv folds.
    :return: arr of DatasetExt instances. Each one corresponds to a sub-dataset.

    ---------
    Notes

    Each sub-dataset its conformed by a training part and a test part.
    Each test part is used in only one sub-dataset.
    """

    sensitive_data = [x_sens.reshape(-1, 1) for x_sens in dataset_ext.sensitive_train.values()]
    data = np.hstack([dataset_ext.x_train] + sensitive_data + [dataset_ext.y_train.reshape((-1, 1))])
    folds = cross_validation_split(data, n_folds)
    dataset_folds = []
    for v in range(len(folds)):
        train_set = list(folds)
        train_set.pop(v)
        train_set = np.array(sum(train_set, []))
        test_set = np.array(folds[v])
        col = dataset_ext.x_train.shape[1]
        x_test, y_test = test_set[:, 0:col], test_set[:, -1]
        x_train, y_train = train_set[:, 0:col], train_set[:, -1]
        sensitive_train = dict()
        sensitive_test = dict()
        for k in dataset_ext.sensitive_train.keys():
            sensitive_train[k] = train_set[:, col]
            sensitive_test[k] = test_set[:, col]
            col += 1
        dataset_ext_v = DatasetExt(x_train=x_train, sensitive_train=sensitive_train, y_train=y_train,
                                   x_test=x_test, sensitive_test=sensitive_test, y_test=y_test,
                                   intercept=dataset_ext.intercept)

        dataset_folds.append(dataset_ext_v)

    return dataset_folds


def construct_in_folds(dataset_folds, max_depth, min_size, get_split, *args):
    """
    In each subdataset of cv, grow the max tree decision in the subdataset.

    --------
    Parameters

    :param dataset_folds: arr of DatasetExt instances. Sub-datasets for cross validation (cv).
    :return: trees_max_folds lists. A sequence of the tree max grow in each sub-dataset.
    """
    v = 1
    trees_max_folds = []
    for dataset_ext_v in dataset_folds:
        print('construct tree max in fold', v)

        if get_split == get_gini_split:
            tree_v = build_tree(max_depth, min_size, dataset_ext_v, get_split)
        else:
            tree_v = build_tree(max_depth, min_size, dataset_ext_v, get_split, *args)
        v += 1
        trees_max_folds.append(tree_v)

    return trees_max_folds


def prun_folds(trees_max_folds, lmbda):
    """
    In each subdataset of cv, grow the max tree decision in the subdataset and
    calculate alphas and trees sequence of optimal complex prunninf algorithm.

    --------
    Parameters

    :param trees_max_folds
    :return: alphas_folds, trees_folds lists of same lenght. The entry number v of the lists are:
    - trees_folds[v]: sequence of optimal prunning subtrees {T_k}^v of the tree max grow in sub-dataset v.
    - alphas_folds[v]: sequence of complexity parameter alpha_{k+1} = g(t_k) of the tree max grow in sub-dataset v.
    """
    alphas_folds, trees_folds = [], []
    v = 1
    for tree_v in trees_max_folds:
        print('pruning fold', v)
        if tree_v['is_terminal']:
            alphas_v, trees_v = [-np.inf], [tree_v]
        else:
            alphas_v, trees_v = minimal_complexity_prunning(tree_v, lmbda)
        alphas_folds.append(alphas_v)
        trees_folds.append(trees_v)
        v += 1

    return alphas_folds, trees_folds


# %%

def count_pred_class(trues, predictions, classes_to_idx, n_classes):
    """
    Calculate missclasification matrix associated with true values and predictions.

    ---------
    Parameters

    :param trues: ndarray shape=(n_test,). True labels of a test dataset.
    :param predictions: ndarray shape=(n_test,). Predictions of a model in the test dataset.
    :param classes_to_idx: dict. Key=Name of classes present in the training dataset,
    value=fixed id.
    :param n_classes: int. Number of total classes present in the training dataset.
    :return: ndarray shape=(n_classes, n_classes). Missclasification matrix.

    --------
    Notes

    The missclasification matrix N, of a tree T has entries defined as:

    N_ij = number of (x,y) in dataset such that y=j and T(x)=i.

    """

    trues = trues.astype('int32')
    predictions = predictions.astype('int32')
    N_pred_clss = np.zeros((n_classes, n_classes))
    for idx, clss in enumerate(trues):
        pred_idx = classes_to_idx[predictions[idx]]
        clss_idx = classes_to_idx[clss]
        N_pred_clss[pred_idx][clss_idx] += 1
    return N_pred_clss


def count_prediction_class_in_folds(trees_folds, dataset_folds, classes_to_idx, n_classes):
    """

    Calculate missclasification matrix for each fold v and each optimal tree of the fold.

    ---------
    Parameters

    :param trees_folds: arr of arr. Array of sequence of optimal prunning subtrees {T_k}^v
        for each sub-dataset v.
    :param dataset_folds: arr of DatasetExt instances. Sub-datasets for cross validation (cv).
    :param n_classes: int. Number of classes in the complete dataset.
    :return: arr of arr. The v entry of the outer array is an arr of ndarray shape=(n_classes, n_classes).
    The k entry of the inner array is the missclasification matrix N^{v,k}  of the k optimal tree T(k)^v
    of the v subdataset.

    """

    N_alphas_folds = []
    for v, dataset_ext_v in enumerate(dataset_folds):
        test_v = dataset_ext_v.x_test
        trues_v = dataset_ext_v.y_test
        trees_v = trees_folds[v]

        N_alphas_v = []
        for tree_alpha_v in trees_v:
            predictions_alpha = prediction_set(tree_alpha_v, test_v)
            N_alpha_v = count_pred_class(trues_v, predictions_alpha, classes_to_idx, n_classes)
            N_alphas_v.append(N_alpha_v)
        N_alphas_folds.append(N_alphas_v)
    return N_alphas_folds


# %%

def count_prediction_class_cv(alphas_geom, alphas_folds, N_alphas_folds, n_classes):
    """
    Calculate misclassification matrix for each optimal tree, tree(alpha), using cv.

    ---------
    Parameters

    :param alphas_geom: arr. The geometric middle points of the alpha sequence of the prunning
        of tree max grow in the full dataset.
    :param alphas_folds: Array of sequence of optimal prunning alphas {alpha_k}^v
        for each sub-dataset v.
    :param N_alphas_folds: Array of v sequences of misclassification matrices {N^{v,k}}k=1^K_v
        for each sub-dataset v and each optimal tree T^v(k).
    :param n_classes: int. Number of classes in the complete dataset.
    :return: array of narray shape=(n_classes, n_classes). Outer array is the same lenght as
    alphas_geom. The narrays are estimate of misclassification matrix with cv.

    --------
    Notes

    tree(alpha) = tree(k) is the k-th tree obtained by prunning a tree max with optimal prunning.
    The estimate of missclasification matrix N(k) with cv is equal to the entrywise sum
    of missclasification matrices N(k)^v for each subdataset v of the cv partition.

    """
    N_alphas = []

    for alpha in alphas_geom:
        N_alpha = np.zeros((n_classes, n_classes))
        for alphas_v, N_alphas_v in zip(alphas_folds, N_alphas_folds):
            # find alpha^v_{l} such that: alpha^v_{l} <= alpha < alpha^v_{l+1}
            idx_alpha, num_alphas_v = 0, len(alphas_v)
            while idx_alpha < num_alphas_v and alphas_v[idx_alpha] <= alpha:
                idx_alpha += 1
            idx_alpha = max(0, idx_alpha - 1)

            # sum N_alpha^v_{l} to N_alpha
            N_alpha += N_alphas_v[idx_alpha]
        N_alphas.append(N_alpha)
    return N_alphas


# %%

def calculate_misclass_cv(N_alphas, dataset_ext):
    """

    Estimate misclassification R(alpha) for each optimal tree, tree(alpha), using cv.

    ---------
    Parameters
    :param N_alphas: array of narray shape=(n_classes, n_classes). Outer array is the same lenght as
    alphas_geom. The narrays are estimate of misclassification matrix with cv.
    :param dataset_ext: DatasetExt instance. The full data with information of target classes (misclassification costs,
    apriori probabilities of classes, and size of classes in full data)
    :return: ndarray shape_like(alphas_geom). Arr with entries R(alpha) for each alpha in alphas_geom.

    ---------
    Notes         # TODO: ¿se puede agregar E^cv(t)?

    R(alpha)= sum_{i, j} C(i | j) N_{i j}/N_j

    """

    R_alphas = []
    for N_alpha in N_alphas:
        Q_alpha = np.apply_along_axis(lambda N: np.divide(N, dataset_ext.classes_sizes), 1, N_alpha)
        R_alpha_k = np.sum(dataset_ext.costs * Q_alpha, axis=0)
        R_alpha = np.dot(R_alpha_k, dataset_ext.aprioris)
        R_alphas.append(R_alpha)
    return np.array(R_alphas)


# %%
def sort_r_alphas(trees, R_trees, alphas, n_terminals, p_rules, n_rules):
    """

    ---------
    Parameters

    param trees:
    param alphas:
    return:
    """

    # Sort from simple trees to complex trees
    alpha_R_trees = zip(alphas, R_trees, trees, n_terminals, p_rules, n_rules)
    alpha_R_trees = sorted(alpha_R_trees, key=lambda tupl: tupl[3], reverse=False)
    alphas = np.array([tup[0] for tup in alpha_R_trees])
    R_trees = np.array([tup[1] for tup in alpha_R_trees])
    trees = [tup[2] for tup in alpha_R_trees]
    n_terminals = [tup[3] for tup in alpha_R_trees]
    p_rules = [tup[4] for tup in alpha_R_trees]
    n_rules = [tup[5] for tup in alpha_R_trees]

    return trees, R_trees, alphas, n_terminals, p_rules, n_rules


def count_predictions_by_sens_groups_in_folds(trees_folds, dataset_folds):
    counts_folds = []
    for dataset_ext_v, trees_v in zip(dataset_folds, trees_folds):
        counts_alphas_fold_v = []
        for tree_alpha_v in trees_v:
            correlation_a_pred_dict_alpha = get_model_correlations(tree_alpha_v, dataset_ext_v, test=True, val=False,
                                                                   percentage=False)
            counts_alphas_fold_v.append(correlation_a_pred_dict_alpha)
        counts_folds.append(counts_alphas_fold_v)
    return counts_folds


def calculate_cv_rules(alphas_geom, alphas_folds, counts_folds, dataset_ext):
    group_0_pred_pos, group_0_pred_neg = np.zeros_like(alphas_geom), np.zeros_like(alphas_geom)
    group_1_pred_pos, group_1_pred_neg = np.zeros_like(alphas_geom), np.zeros_like(alphas_geom)

    M = max(alphas_geom)
    for idx_alpha, alpha in enumerate(alphas_geom):
        for alphas_v, counts_v in zip(alphas_folds, counts_folds):
            # find corresponding alpha in fold v, i.e. find alpha^v_{l}
            if alpha == M:
                idx_alpha_v = np.argmax(alphas_v)
            else:
                # such that: alpha^v_{l} <= alpha < alpha^v_{l+1}
                idx_alpha_v, num_alphas_v = 0, len(alphas_v)
                while idx_alpha_v < num_alphas_v and alphas_v[idx_alpha_v] <= alpha:
                    idx_alpha_v += 1
                idx_alpha_v = max(0, idx_alpha_v - 1)

            # sum counts conjunction 'a' and 'pred' of corresponding alpha in fold v
            group_0_pred_pos[idx_alpha] += counts_v[idx_alpha_v][0][1]
            group_0_pred_neg[idx_alpha] += counts_v[idx_alpha_v][0][-1]
            group_1_pred_pos[idx_alpha] += counts_v[idx_alpha_v][1][1]
            group_1_pred_neg[idx_alpha] += counts_v[idx_alpha_v][1][-1]

    sens_to_idx = dataset_ext.sensitive_groups_to_idx
    sens_sizes = dataset_ext.sensitive_sizes

    group_0_pred_pos = group_0_pred_pos/float(sens_sizes[sens_to_idx[0]])
    group_0_pred_neg = group_0_pred_neg/float(sens_sizes[sens_to_idx[0]])
    group_1_pred_pos = group_1_pred_pos/float(sens_sizes[sens_to_idx[1]])
    group_1_pred_neg = group_1_pred_neg/float(sens_sizes[sens_to_idx[1]])

    p_rules, n_rules = np.zeros_like(alphas_geom), np.zeros_like(alphas_geom)

    for idx_alpha in range(len(alphas_geom)):
        p_rule, n_rule = p_rules_model_stats(non_prot_pos=group_1_pred_pos[idx_alpha],
                                             prot_pos=group_0_pred_pos[idx_alpha],
                                             non_prot_neg=group_1_pred_neg[idx_alpha],
                                             prot_neg=group_0_pred_neg[idx_alpha],
                                             multiply=1.0)
        p_rules[idx_alpha] = p_rule
        n_rules[idx_alpha] = n_rule

    return p_rules, n_rules


def get_cv_decision_trees(dataset_ext, dataset_folds, tree, trees_max_folds, lmbda):
    """
    Construct a list of optimal prunning trees in a dataset and calculate their misclassification
    estimates with cross validation (cv).

    ---------
    Parameters

    :param dataset_ext: DatasetExt instance. The full data.
    :param n_folds: int. Number of cv folds.
    :param max_depth: Maximum depth of maximum trees.
    :param min_size: Minimum number of training cases in a node.
    :return: [trees, R_alphas, scores]:
    - trees: list of optimal prunning trees, T(alpha), of the tree max constructed in full dataset.
    - R_alphas: cv estimate o misclassification of each T(alpha).
    - scores: accuracy of each T(alpha) in the train dataset.
    """

    alphas, trees = minimal_complexity_prunning(tree, lmbda)

    # Construction and prunnning of trees in folds of data
    print('Prunnning of trees in folds of data')
    alphas_folds, trees_folds = prun_folds(trees_max_folds, lmbda)

    # Calculate RCV(T(alpha))
    alphas_geom = alphas

    # calculate N^{v,\alpha} for each fold v and T^v(\alpha)
    n_classes = dataset_ext.n_classes
    classes_to_idx = dataset_ext.classes_to_idx
    N_alphas_folds = count_prediction_class_in_folds(trees_folds, dataset_folds, classes_to_idx, n_classes)
    # calculate N^\alpha
    N_alphas = count_prediction_class_cv(alphas_geom, alphas_folds, N_alphas_folds, n_classes)
    # calculate RCV(T(alpha))
    R_trees = calculate_misclass_cv(N_alphas, dataset_ext)

    # Calculate p_rule^CV(T(alpha)) and n_rule^CV(T(alpha))
    counts_folds = count_predictions_by_sens_groups_in_folds(trees_folds, dataset_folds)
    p_rules, n_rules = calculate_cv_rules(alphas_geom, alphas_folds, counts_folds, dataset_ext)

    # order trees by complexity (trivial to full)
    for tr in trees:
        calculate_terminals_of_node(tr)
    n_terminals = np.array([tr['n_terminals'] for tr in trees])
    trees, R_trees, alphas, n_terminals, p_rules, n_rules = sort_r_alphas(trees, R_trees, alphas, n_terminals, p_rules, n_rules)

    return trees, np.array(R_trees), alphas, n_terminals, np.array(p_rules), np.array(n_rules)
