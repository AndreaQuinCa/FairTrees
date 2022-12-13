import pandas as pd

from evaluations_fairness import *
from visualization_trees import draw_tree
from tree_construction import calculate_terminals_of_node


def add_path(filename, my_path=None):
    if my_path:
        filename = my_path + '/' + filename
    return filename

def get_report_col_names(continous_idx_set, max_depth, fmccp=False):
    report_column_names = ["Runtime", "Accuracy", "Depth",
                           "p-value (protected)", "p-rule", "n-rule",
                           "nreps", "cov", "lambda", "p-epsilon", "constant", "n terminals",
                           "SP", "B. accuracy", "B. accuracy prot.", "B. accuracy non-prot."]

    metrics_depths_col_names = None
    if max_depth is not None:
        metrics_depths_col_names = ["Accuracy %d,p-value %d,n-value %d" % tuple([depth] * 3) for depth in
                                    np.arange(max_depth + 1)]

        metrics_depths_col_names = ",".join(metrics_depths_col_names).split(',')

    if fmccp:
        r_prune_col_names = ["R %d" % alpha for alpha in np.arange(100)]
        alpha_prune_col_names = ["alpha %d" % alpha for alpha in np.arange(100)]
        prules_prune_col_names = ["alpha %d" % alpha for alpha in np.arange(100)]
        nrules_prune_col_names = ["alpha %d" % alpha for alpha in np.arange(100)]
    else:
        r_prune_col_names, alpha_prune_col_names, prules_prune_col_names, nrules_prune_col_names = None, None, None, None

    variable_types_depths_col_names = None
    if continous_idx_set and max_depth is not None:
        variable_types_depths_col_names = [
            "Discrete %d,Continuous %d,Cases in discrete %d, Cases in continuous %d" % tuple([depth] * 4) for depth in
            np.arange(max_depth + 1)]
        variable_types_depths_col_names = ",".join(variable_types_depths_col_names).split(',')

    return {'report_column_names': report_column_names, 'metrics_depths_col_names': metrics_depths_col_names,
            'variable_types_depths_col_names': variable_types_depths_col_names,
            'r_prune_col_names': r_prune_col_names, 'alpha_prune_col_names': alpha_prune_col_names,
            'prules_prune_col_names': prules_prune_col_names, 'nrules_prune_col_names': nrules_prune_col_names}


def get_result_containers(n_reps, column_names):
    r_prune_col_names, alpha_prune_col_names = column_names['r_prune_col_names'], column_names['alpha_prune_col_names']
    report_column_names, metrics_depths_col_names = column_names['report_column_names'], column_names[
        'metrics_depths_col_names']
    variable_types_depths_col_names = column_names['variable_types_depths_col_names']

    report_mat = np.zeros((n_reps, len(report_column_names)))
    report_mat_val = np.zeros((n_reps, len(report_column_names)))
    report_mat_train = np.zeros((n_reps, len(report_column_names)))
    metrics_depths_mat, metrics_depths_mat_train = None, None
    if metrics_depths_col_names is not None:
        metrics_depths_mat = np.zeros((n_reps, len(metrics_depths_col_names))) - 1
        metrics_depths_mat_train = np.zeros((n_reps, len(metrics_depths_col_names))) - 1
    r_prune_mat, p_rules_mat, n_rules_mat, alpha_prune_mat = None, None, None, None
    if r_prune_col_names is not None:
        r_prune_mat = np.zeros((n_reps, len(r_prune_col_names))) - 1
        alpha_prune_mat = np.zeros((n_reps, len(alpha_prune_col_names))) - 1
        p_rules_mat = np.zeros((n_reps, len(alpha_prune_col_names))) - 1
        n_rules_mat = np.zeros((n_reps, len(alpha_prune_col_names))) - 1
    variable_types_depths_mat = None
    if variable_types_depths_col_names is not None:
        variable_types_depths_mat = np.zeros((n_reps, len(variable_types_depths_col_names))) - 1
    return {'report_mat': report_mat, 'report_mat_val': report_mat_val, 'report_mat_train': report_mat_train,
            'metrics_depths_mat': metrics_depths_mat, 'metrics_depths_mat_train': metrics_depths_mat_train,
            'variable_types_depths_mat': variable_types_depths_mat,
            'r_prune_mat': r_prune_mat, 'p_rules_mat': p_rules_mat, 'n_rules_mat': n_rules_mat,
            'alpha_prune_mat': alpha_prune_mat}


def write_results(model, rep, n_reps, runtime, cov, lmbda, p_epsilon, dataset_ext, rs_prune, p_rules, n_rules,
                  alphas_prune, containers, continous_idx_set, is_tree=True):
    report_mat, report_mat_val, report_mat_train = containers['report_mat'], containers['report_mat_val'], containers[
        'report_mat_train']
    metrics_depths_mat, metrics_depths_mat_train = containers['metrics_depths_mat'], containers[
        'metrics_depths_mat_train']
    r_prune_mat, alpha_prune_mat = containers['r_prune_mat'], containers['alpha_prune_mat']
    p_rules_mat, n_rules_mat = containers['p_rules_mat'], containers['n_rules_mat']
    variable_types_depths_mat = containers['variable_types_depths_mat']

    # model metrics test
    report_mat[rep, 0] = round(runtime, 2)
    preds, trues = get_tree_predictions_and_trues(model, dataset_ext, test=True, val=False, is_tree=is_tree)
    s_attr_name = dataset_ext.sensitive_attrs[0]
    sensitive = dataset_ext.sensitive_test[s_attr_name]
    sp, accuracy = calculate_performance_sp(trues, preds, sensitive)
    report_mat[rep, 1] = round(accuracy, 4)  # accuracy test
    depth = np.nan
    if is_tree:
        depth = get_depth(model)
    report_mat[rep, 2] = depth


    # fair metrics test
    correlation_dict_test = get_model_correlations(model, dataset_ext, test=True, is_tree=is_tree)
    p_prot_value = get_model_one_p_value(correlation_dict_test, s_attr_name)
    report_mat[rep, 3] = round(p_prot_value, 4)  # p protected value
    p_value, n_value = p_rules_model_stats(correlation_dict_test, s_attr_name)
    report_mat[rep, 4] = round(p_value, 4)  # p value
    report_mat[rep, 5] = round(n_value, 4)  # n value
    report_mat[rep, 12] = round(sp, 4)
    cm, cm_prot, cm_non_prot = get_confusion_matrices(trues, preds, sensitive)
    b_acc = get_balanced_accuracy(cm)
    b_acc_prot = get_balanced_accuracy(cm_prot)
    b_acc_non_prot = get_balanced_accuracy(cm_non_prot)
    report_mat[rep, 13] = round(b_acc, 4)
    report_mat[rep, 14] = round(b_acc_prot, 4)
    report_mat[rep, 15] = round(b_acc_non_prot, 4)

    # experiment details
    report_mat[rep, 6] = n_reps
    report_mat[rep, 7] = cov
    report_mat[rep, 8] = lmbda
    report_mat[rep, 9] = p_epsilon
    if is_tree:
        report_mat[rep, 10] = is_constant(model)
        calculate_terminals_of_node(model)
        report_mat[rep, 11] = model['n_terminals']
    else:
        report_mat[rep, 10] = np.nan
        report_mat[rep, 11] = np.nan


    # tree metrics val
    report_mat_val[rep, 0] = round(runtime, 2)
    preds, trues = get_tree_predictions_and_trues(model, dataset_ext, test=False, val=True, is_tree=is_tree)
    sensitive = dataset_ext.sensitive_val[s_attr_name]
    sp, accuracy = calculate_performance_sp(trues, preds, sensitive)
    report_mat_val[rep, 1] = round(accuracy, 4)  # accuracy test
    if is_tree:
        report_mat_val[rep, 2] = depth

    # fair metrics val
    correlation_dict_val = get_model_correlations(model, dataset_ext, test=False, val=True, is_tree=is_tree)
    p_prot_value = get_model_one_p_value(correlation_dict_val, s_attr_name)
    report_mat_val[rep, 3] = round(p_prot_value, 4)  # p protected value
    p_value, n_value = p_rules_model_stats(correlation_dict_val, s_attr_name)
    report_mat_val[rep, 4] = round(p_value, 4)  # p value
    report_mat_val[rep, 5] = round(n_value, 4)  # n value
    report_mat_val[rep, 12] = round(sp, 4)
    cm, cm_prot, cm_non_prot = get_confusion_matrices(trues, preds, sensitive)
    b_acc = get_balanced_accuracy(cm)
    b_acc_prot = get_balanced_accuracy(cm_prot)
    b_acc_non_prot = get_balanced_accuracy(cm_non_prot)
    report_mat_val[rep, 13] = round(b_acc, 4)
    report_mat_val[rep, 14] = round(b_acc_prot, 4)
    report_mat_val[rep, 15] = round(b_acc_non_prot, 4)

    # experiment details
    report_mat_val[rep, 6] = n_reps
    report_mat_val[rep, 7] = cov
    report_mat_val[rep, 8] = lmbda
    report_mat_val[rep, 9] = p_epsilon
    if is_tree:
        report_mat_val[rep, 10] = is_constant(model)
        report_mat_val[rep, 11] = model['n_terminals']
    else:
        report_mat_val[rep, 10] = np.nan
        report_mat_val[rep, 11] = np.nan

    # tree metrics train
    report_mat_train[rep, 0] = round(runtime, 2)
    preds, trues = get_tree_predictions_and_trues(model, dataset_ext, test=False, val=False, is_tree=is_tree)
    sensitive = dataset_ext.sensitive_train[s_attr_name]
    sp, accuracy = calculate_performance_sp(trues, preds, sensitive)
    report_mat_train[rep, 1] = round(accuracy, 4)
    if is_tree:
        report_mat_train[rep, 2] = depth

    # fair metrics train
    correlation_dict_train = get_model_correlations(model, dataset_ext, test=False, val=False, is_tree=is_tree)
    s_attr_name = dataset_ext.sensitive_attrs[0]
    assert (s_attr_name == dataset_ext.sensitive_attrs[0])
    p_prot_value = get_model_one_p_value(correlation_dict_train, s_attr_name)
    report_mat_train[rep, 3] = round(p_prot_value, 4)  # p protected value
    p_value, n_value = p_rules_model_stats(correlation_dict_train, s_attr_name)
    report_mat_train[rep, 4] = round(p_value, 4)  # p value
    report_mat_train[rep, 5] = round(n_value, 4)  # n value
    report_mat_train[rep, 12] = round(sp, 4)
    cm, cm_prot, cm_non_prot = get_confusion_matrices(trues, preds, sensitive)
    b_acc = get_balanced_accuracy(cm)
    b_acc_prot = get_balanced_accuracy(cm_prot)
    b_acc_non_prot = get_balanced_accuracy(cm_non_prot)
    report_mat_train[rep, 13] = round(b_acc, 4)
    report_mat_train[rep, 14] = round(b_acc_prot, 4)
    report_mat_train[rep, 15] = round(b_acc_non_prot, 4)

    # experiment details
    report_mat_train[rep, 6] = n_reps
    report_mat_train[rep, 7] = cov
    report_mat_train[rep, 8] = lmbda
    report_mat_train[rep, 9] = p_epsilon
    if is_tree:
        report_mat_train[rep, 10] = is_constant(model)
        report_mat_train[rep, 11] = model['n_terminals']
    else:
        report_mat_train[rep, 10] = np.nan
        report_mat_train[rep, 11] = np.nan

    # metrics by depth
    if is_tree:
        col = 0
        for depth_idx in np.arange(depth + 1):
            # test
            acc_depth, p_value_depth, n_value_depth = get_metrics_of_depth(model, dataset_ext, depth_idx, test=True)
            metrics_depths_mat[rep, col + 0] = np.round(acc_depth, 2)
            metrics_depths_mat[rep, col + 1] = np.round(p_value_depth, 2)
            metrics_depths_mat[rep, col + 2] = np.round(n_value_depth, 2)

            # train
            acc_depth, p_value_depth, n_value_depth = get_metrics_of_depth(model, dataset_ext, depth_idx, test=False)
            metrics_depths_mat_train[rep, col + 0] = np.round(acc_depth, 2)
            metrics_depths_mat_train[rep, col + 1] = np.round(p_value_depth, 2)
            metrics_depths_mat_train[rep, col + 2] = np.round(n_value_depth, 2)
            col += 3

    # variable types
    if continous_idx_set and is_tree:
        col = 0
        for depth_idx in np.arange(depth + 1):
            n_discrete, n_continuous, n_cases_dis, n_cases_con = count_variable_types(model, depth_idx,
                                                                                      continous_idx_set,
                                                                                      dataset_ext.training_size)
            variable_info = [n_discrete, n_continuous, n_cases_dis, n_cases_con]
            for info_idx, info in enumerate(variable_info):
                variable_types_depths_mat[rep, col + info_idx] = info
            col += 4

    # R^cv
    if rs_prune is not None and alphas_prune is not None:
        for col in range(len(rs_prune)):
            if col >= r_prune_mat.shape[1]:
                print('rs_prune in rep %d exceeds r_prune_mat cols:' % rep)
                print('rs_prune size: %d' % len(rs_prune))
                break
            r_prune_mat[rep, col] = rs_prune[col]

        for col in range(len(alphas_prune)):
            if col >= alpha_prune_mat.shape[1]:
                print('alphas_prune in rep %d exceeds alpha_prune_mat cols:' % rep)
                print('alphas_prune size: %d' % len(alphas_prune))
                break
            alpha_prune_mat[rep, col] = alphas_prune[col]

    if p_rules is not None and n_rules is not None:
        for col in range(len(p_rules)):
            if col >= p_rules_mat.shape[1]:
                print('p_rules in rep %d exceeds p_rules_mat cols:' % rep)
                print('p_rules size: %d' % len(p_rules))
                break
            p_rules_mat[rep, col] = p_rules[col]
        for col in range(len(n_rules)):
            if col >= n_rules_mat.shape[1]:
                print('p_rules in rep %d exceeds p_rules_mat cols:' % rep)
                print('p_rules size: %d' % len(n_rules))
                break
            n_rules_mat[rep, col] = n_rules[col]


def save_results(containers, column_names, continous_idx_set,
                 lmbda, cov, p_epsilon, n_reps, method_name, data_name, my_path,
                 prune_method=None, constr_method=None, experiment_variation=None):

    r_prune_col_names, alpha_prune_col_names = column_names['r_prune_col_names'], column_names['alpha_prune_col_names']
    prules_prune_col_names, nrules_prune_col_names = column_names['prules_prune_col_names'], column_names[
        'nrules_prune_col_names']
    report_column_names, metrics_depths_col_names = column_names['report_column_names'], column_names[
        'metrics_depths_col_names']
    variable_types_depths_col_names = column_names['variable_types_depths_col_names']

    report_mat, report_mat_val, report_mat_train = containers['report_mat'], containers['report_mat_val'], containers[
        'report_mat_train']
    metrics_depths_mat, metrics_depths_mat_train = containers['metrics_depths_mat'], containers[
        'metrics_depths_mat_train']
    r_prune_mat, alpha_prune_mat = containers['r_prune_mat'], containers['alpha_prune_mat']
    p_rules_mat, n_rules_mat = containers['p_rules_mat'], containers['n_rules_mat']
    variable_types_depths_mat = containers['variable_types_depths_mat']

    if lmbda is not None:
        if cov is not None:
            exp_details = [('cov %0.5f', cov), ('lambda %0.2f', lmbda)]
        else:
            exp_details = [('lambda %0.2f', lmbda)]
    elif p_epsilon is not None:
        if cov is not None:
            exp_details = [('cov %0.5f', cov), ('p_epsilon %0.2f', p_epsilon)]
        else:
            exp_details = [('p_epsilon %0.2f', p_epsilon)]
    elif cov is not None:
        exp_details = [('cov %0.5f', cov)]
    else:
        exp_details = []
    exp_details = [('reps %d', n_reps)] + exp_details
    exp_details = [k % v for k, v in exp_details]
    exp_details = ' '.join(exp_details)
    experiment_name = ' '.join([method_name, data_name, exp_details])

    # Write results
    if prune_method is not None and constr_method is not None:
        experiments_folder = '/'.join(['tables', 'experiments ' + prune_method, constr_method, data_name])
    else:
        experiments_folder = '/'.join(['tables', 'experiments ' + method_name, data_name])

    metrics = 'metrics'
    compare_metrics_folder_2 = None
    if 'Gini' in constr_method:
        if prune_method == 'MCCP' and constr_method == 'Gini':
            compare_metrics_folder = '/'.join(['tables', 'to_compare_Gini Gini+MCCP', data_name, metrics])
        elif prune_method == 'FMCCP' and constr_method == 'Gini':
            compare_metrics_folder = '/'.join(['tables', 'to_compare_Gini Gini+FMCCP', data_name, metrics])
        elif prune_method == 'RELAB' and constr_method == 'Gini':
            compare_metrics_folder = '/'.join(['tables', 'to_compare_Gini Gini+RELAB', data_name, metrics])
        else:
            assert prune_method is None
            compare_metrics_folder = '/'.join(['tables', 'to_compare_Gini Gini+MCCP', data_name, metrics])
            compare_metrics_folder_2 = '/'.join(['tables', 'to_compare_Gini Gini+FMCCP', data_name, metrics])
    else:
        if 'LRT' in constr_method:
            compare_metrics_folder = '/'.join(['tables', 'to_compare_LRT_C-LRT', data_name, metrics])
        elif 'LR' in constr_method:
            compare_metrics_folder = '/'.join(['tables', 'to_compare_LR_C-LR', data_name, metrics])

    experiments_folder = add_path(experiments_folder, my_path)
    compare_metrics_folder = add_path(compare_metrics_folder, my_path)
    if compare_metrics_folder_2 is not None:
        compare_metrics_folder_2 = add_path(compare_metrics_folder_2, my_path)

    general_metrics_test = 'general metrics test'
    general_metrics_val = 'general metrics val'
    prune_results = 'prune results'
    if experiment_variation is not None:
        experiment_name = experiment_name + ' ' + experiment_variation

    # metrics tests
    report_mat_df = pd.DataFrame(data=report_mat)
    report_mat_df.columns = report_column_names
    report_mat_df['Method'] = [method_name] * report_mat.shape[0]
    report_mat_df['Set'] = ['test'] * report_mat.shape[0]
    path = '/'.join([experiments_folder, general_metrics_test, experiment_name + ' test.csv'])
    report_mat_df.to_csv(path, index=False)
    path = '/'.join([compare_metrics_folder, experiment_name + ' test.csv'])
    report_mat_df.to_csv(path, index=False)
    if compare_metrics_folder_2 is not None:
        path = '/'.join([compare_metrics_folder_2, experiment_name + ' test.csv'])
        report_mat_df.to_csv(path, index=False)

    # metrics val
    report_mat_df = pd.DataFrame(data=report_mat_val)
    report_mat_df.columns = report_column_names
    report_mat_df['Method'] = [method_name] * report_mat.shape[0]
    report_mat_df['Set'] = ['val'] * report_mat.shape[0]
    path = '/'.join([experiments_folder, general_metrics_val, experiment_name + ' val.csv'])
    print(report_mat_df.dropna(axis=1))
    report_mat_df.to_csv(path, index=False)



    # metrics train
    report_mat_df = pd.DataFrame(data=report_mat_train)
    report_mat_df.columns = report_column_names
    report_mat_df['Method'] = [method_name] * report_mat.shape[0]
    report_mat_df['Set'] = ['train'] * report_mat.shape[0]
    report_mat_df.to_csv(experiments_folder + '/' + experiment_name + ' train.csv', index=False)

    # metrics by depth test
    if metrics_depths_mat is not None:
        metrics_depths_mat = pd.DataFrame(data=metrics_depths_mat)
        metrics_depths_mat.columns = metrics_depths_col_names
        metrics_depths_mat['Method'] = [method_name] * metrics_depths_mat.shape[0]
        metrics_depths_mat['Set'] = ['test'] * metrics_depths_mat.shape[0]
        metrics_depths_mat.to_csv(experiments_folder + '/' + experiment_name + ' test by depth.csv', index=False)

        # metrics by depth train
        metrics_depths_mat = pd.DataFrame(data=metrics_depths_mat_train)
        metrics_depths_mat.columns = metrics_depths_col_names
        metrics_depths_mat['Method'] = [method_name] * metrics_depths_mat.shape[0]
        metrics_depths_mat['Set'] = ['train'] * metrics_depths_mat.shape[0]
        metrics_depths_mat.to_csv(experiments_folder + '/' + experiment_name + ' train by depth.csv', index=False)

    # variable type counts by depth
    if continous_idx_set and variable_types_depths_mat is not None:
        variable_types_depths_mat = pd.DataFrame(data=variable_types_depths_mat)
        variable_types_depths_mat.columns = variable_types_depths_col_names
        variable_types_depths_mat['Method'] = [method_name] * variable_types_depths_mat.shape[0]
        variable_types_depths_mat.to_csv(experiments_folder + '/' + experiment_name + ' variable type by depth.csv',
                                         index=False)

    # cv prunning results
    if r_prune_mat is not None:
        r_prune_df = pd.DataFrame(data=r_prune_mat)
        r_prune_df.columns = r_prune_col_names
        r_prune_df['Method'] = [method_name] * r_prune_df.shape[0]
        r_prune_df.to_csv('/'.join([experiments_folder, prune_results, experiment_name + ' r.csv']), index=False)

        alpha_prune_df = pd.DataFrame(data=alpha_prune_mat)
        alpha_prune_df.columns = alpha_prune_col_names
        alpha_prune_df['Method'] = [method_name] * alpha_prune_df.shape[0]
        alpha_prune_df.to_csv('/'.join([experiments_folder, prune_results, experiment_name + ' alpha.csv']),
                              index=False)

    if p_rules_mat is not None:
        p_rules_df = pd.DataFrame(data=p_rules_mat)
        p_rules_df.columns = prules_prune_col_names
        p_rules_df['Method'] = [method_name] * p_rules_df.shape[0]
        p_rules_df.to_csv('/'.join([experiments_folder, prune_results, experiment_name + ' prule.csv']), index=False)

        n_rules_df = pd.DataFrame(data=n_rules_mat)
        n_rules_df.columns = nrules_prune_col_names
        n_rules_df['Method'] = [method_name] * n_rules_df.shape[0]
        n_rules_df.to_csv('/'.join([experiments_folder, prune_results, experiment_name + ' nrule.csv']), index=False)


def save_tree(tree, p_epsilon, n_rep, method_name, data_name):
    if p_epsilon is not None:
        exp_details = [('p_epsilon %0.2f', p_epsilon), ('rep %d', n_rep)]
    else:
        exp_details = [('rep %d', n_rep)]
    exp_details = [k % v for k, v in exp_details]
    exp_details = ' '.join(exp_details)
    experiment_name = ' '.join([method_name, data_name, exp_details])
    draw_tree(tree, experiment_name)
