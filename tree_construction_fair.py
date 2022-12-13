# coding=utf-8
# %%
from tree_optimization_fair_constrains import *
from tree_construction import *
from visualization_trees import plot_node


# %%

# Select the best cut for a predictor
def get_optimal_cut(pred_idx, group, loss_function, apply_fairness_constraints, apply_accuracy_constraint,
                    sep_constraint,
                    sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma):
    """
    :param pred_idx: the predictor that will be splited
    :param group: DatasetExt instance. Data for optimization.
    :param loss_function:
    :param apply_fairness_constraints: bool.
    :param apply_accuracy_constraint: bool.
    :param sep_constraint:  bool.
    :param sensitive_attrs: arr.
    :param sensitive_attrs_to_cov_thresh: dict.
    :param gamma: float.
    :return: w: the best cut and its accuracy in (group.x_test, group.y_test).

    """
    x_i_train = group.x_train[:, [0, pred_idx]]

    w, one_sensible_class, w_unconstrained = train_model(x_i_train, group.y_train, group.sensitive_train, loss_function,
                                                         apply_fairness_constraints,
                                                         apply_accuracy_constraint, sep_constraint, sensitive_attrs,
                                                         sensitive_attrs_to_cov_thresh,
                                                         gamma)

    # Accuracy report
    test_score = get_split_accuracy(pred_idx, w, group.x_train, group.y_train)
    cut = -w[0] / w[1]

    return w, cut, test_score, one_sensible_class, w_unconstrained


# Select the best split
def get_fair_split(group, depth, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
                   sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma, plot=False):
    acc_best, pred_idx_best, w_best, cut_best, groups_best, p_best, n_best, cov_best, w_u_best = -1.0, None, None, None, None, \
                                                                                                 None, None, None, None

    sensitive_attrs_to_constrain = sensitive_attrs_to_cov_thresh

    for pred_idx in group.predictors_idx:
        w, cut, acc, one_sensible_class, w_unconstrained = get_optimal_cut(pred_idx, group, loss_function,
                                                                           apply_fairness_constraints,
                                                                           apply_accuracy_constraint, sep_constraint,
                                                                           sensitive_attrs,
                                                                           sensitive_attrs_to_constrain, gamma)

        if acc > acc_best:
            acc_best, pred_idx_best, cut_best, w_best, w_u_best = acc, pred_idx, cut, w, w_unconstrained

    groups_best = split_cases_by_cut(pred_idx_best, cut_best, group)
    R_best = None
    R_fair = None
    plot = False
    pred = None
    acc = None

    if groups_best is not None:
        p_best, n_best, cov_best = get_node_fair_statistics(group, pred_idx_best, w_best, sensitive_attrs, test=False)
        R_best, pred = estimate_mis_node(group)
        acc = get_group_acc(group, pred, test=False)
        R_fair = 1.-estimate_rule_node(group)
        assert 0 <= R_fair <= 1.
        cut_best_u = -w_u_best[0] / w_u_best[1]
        if plot and abs(cut_best_u-cut_best) > 1e-4:
            plot_node(group, pred_idx_best, w_u_best, sensitive_attrs, cut_best, p_best, acc_best, depth)

    return {'index': pred_idx_best, 'cut': cut_best, 'group': group, 'groups': groups_best,
            'R_node': R_best, 'R_fair': R_fair, 'p_value': p_best, 'n_value': n_best,
            'mean_cov': cov_best, 'pred': pred, 'acc': acc, 'w': w_best, 'depth': depth}



