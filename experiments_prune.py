# coding=utf-8
# %%
import time
import pickle
from random import seed

from tree_prunning_cv import *
from data_loading_new import prepare_data, get_parameters, format_parameters
from data_saving import *
from tree_construction_fair import get_fair_split
from tree_construction_gini import get_gini_split
from tree_loss_functions import logistic_loss
from tree_relabeling import *


# %%

# Parameters

# data
data_name = 'adult'  # I
continuous = True  # I
if continuous:
    filename_parameters = 'parameters_experiments_' + data_name + '_continuous' + '_fmccp_gini.csv'  # I
    data_name = data_name + ' continuous'
else:
    filename_parameters = 'parameters_experiments_' + data_name + '_fmccp_gini.csv'  # I

my_path = None
test_size = 0.2
val_size = 0.2

# tree
construct = False  # I
n_folds = 7  # I
n_reps = 15  # I
max_depth = 11  # I
get_split = get_gini_split  # I
min_size = 5
draw = False

# seeds
SEED = 110979
seed(SEED)
np.random.seed(SEED)

# fairness
apply_fairness_constraints = 0
loss_function = logistic_loss
apply_accuracy_constraint = 0
sep_constraint = 0
gamma = None
p_epsilon = None
lmbda = None
# %%

# load and prepare parameters and data
if get_split == get_gini_split:
    added_intercept = False
else:
    added_intercept = True

parameters_experiments = get_parameters(filename_parameters, data_name, my_path)

x, y, sensitive, continous_idx_set, s_attr, sensitive_attrs, n_sample = \
    prepare_data(data_name, my_path, added_intercept)

dictionaries_folder = add_path('dictionaries/', my_path)

# report structure
column_names = get_report_col_names(continous_idx_set, max_depth, prune=False)

# %%

if get_split == get_gini_split:
    constr_method = 'Gini'
    method_name = 'Gini'
else:
    constr_method = 'TLR'
    method_name = 'TLR'

if apply_fairness_constraints:
    constr_method = 'C-' + constr_method
    if data_name == 'adult':
        cov = 0.1
    elif data_name == 'compas':
        cov = 1e-5
    elif data_name == 'ricci':
        cov = 1.5
    elif data_name == 'law':
        cov = 0.3
    elif data_name == 'adult continuous':
        cov = .99
    elif data_name == 'compas continuous':
        cov = 0.0005
else:
    assert 'C-' not in constr_method
    cov = None

sensitive_attrs_to_cov_thresh = {s_attr: cov}

tree_names, tree_seeds = [], []
for rep in range(n_reps):
    if cov is not None:
        tree_name = [constr_method, data_name, 'cov %0.04f' % cov, 'm. depth %0.f' % max_depth, 'tree %s.pkl' % rep]
    else:
        tree_name = [constr_method, data_name, 'm. depth %0.f' % max_depth, 'tree %s.pkl' % rep]
    tree_name = ' '.join(tree_name)
    tree_names.append(tree_name)
    tree_seeds.append(SEED * (rep + 1))

# %%
# construct trees
if construct:
    print(constr_method)

    # for saving results
    containers = get_result_containers(n_reps, column_names)

    for rep in range(n_reps):
        print(" N. repetition = %d " % rep)

        # select a subset of the dataset
        m_seed = tree_seeds[rep]

        # get sample
        x_sample, y_sample, sensitive_sample = get_sample(x, y, sensitive, m_seed, n_sample, do_shuffle=True)

        # split into train and test
        x_train, x_val, x_test, sensitive_train, sensitive_val, sensitive_test, y_train, y_val, y_test = \
            split_data(x_sample, sensitive_sample, y_sample, s_attr, val_size=val_size, test_size=test_size)

        dataset_ext = DatasetExt(x_train=x_train, y_train=y_train, sensitive_train=sensitive_train,
                                 x_val=x_val, y_val=y_val, sensitive_val=sensitive_val,
                                 x_test=x_test, y_test=y_test, sensitive_test=sensitive_test,
                                 intercept=added_intercept)

        # build the tree
        start_time = time.time()
        if get_split == get_gini_split:
            tree = build_tree(max_depth, min_size, dataset_ext, get_split)
        else:
            tree = build_tree(max_depth, min_size, dataset_ext, get_split, loss_function,
                              apply_fairness_constraints,
                              apply_accuracy_constraint,
                              sep_constraint,
                              sensitive_attrs,
                              sensitive_attrs_to_cov_thresh, gamma)
        runtime = time.time() - start_time

        # write results
        R_trees, p_rules, n_rules, alphas_prune, = None, None, None, None
        write_results(tree, rep, n_reps, runtime, cov, lmbda, p_epsilon, dataset_ext, R_trees, p_rules, n_rules, alphas_prune,
                      containers, continous_idx_set)

        # save the tree |o|
        tree_name = tree_names[rep]

        with open(dictionaries_folder + tree_name, 'wb') as f:
            pickle.dump(tree, f)

        # Construction of data folds
        dataset_folds = get_dataset_folds(dataset_ext, n_folds)

        # Construction of max trees in data folds
        print('Construction of max trees in data folds')
        trees_max_folds = construct_in_folds(dataset_folds, max_depth, min_size, get_split, loss_function,
                                             apply_fairness_constraints,
                                             apply_accuracy_constraint,
                                             sep_constraint,
                                             sensitive_attrs,
                                             sensitive_attrs_to_cov_thresh, gamma)

        for v, tree_fold_v in enumerate(trees_max_folds):
            tree_name_v = tree_name[:-4] + ' fold %d' % v + '.pkl'
            with open(dictionaries_folder + tree_name_v, 'wb') as f:
                pickle.dump(tree_fold_v, f)

    save_results(containers,
                 column_names, continous_idx_set,
                 lmbda, cov, p_epsilon, n_reps, method_name, data_name, my_path,
                 prune_method=None, constr_method=constr_method, experiment_variation=None)


# %%

# report structure
column_names = get_report_col_names(continous_idx_set, max_depth, prune=True)

# make prune experiments
for idx, parameter in parameters_experiments.iterrows():

    # get experiment paremeters
    method_name, apply_fairness_constraints, sensitive_attrs_to_cov_thresh, cov, mccp, fmccp, relab, lmbda, p_epsilon = \
        format_parameters(parameter, s_attr)
    assert constr_method == parameter['constr_method']
    prune_method = parameter['prune_method']
    assert prune_method in method_name

    # for saving results
    containers = get_result_containers(n_reps, column_names)
    containers_acc_n_fair = get_result_containers(n_reps, column_names)

    # load Decisions Trees in n_reps samples and prune it
    for rep in range(n_reps):
        print("\n N. repetition = %d " % rep)

        # load the max tree (growed in whole training data)
        tree_name = tree_names[rep]
        with open(dictionaries_folder + tree_name, 'rb') as f:
            tree_max = pickle.load(f)

        # load the max tree (growed in fold of training data)
        trees_max_folds = []
        for v in range(n_folds):
            tree_name_v = tree_name[:-4] + ' fold %d' % v + '.pkl'
            with open(dictionaries_folder + tree_name_v, 'rb') as f:
                tree_max_fold_v = pickle.load(f)
                trees_max_folds.append(tree_max_fold_v)

        # select the subset of the dataset
        m_seed = tree_seeds[rep]

        # get sample
        x_sample, y_sample, sensitive_sample = get_sample(x, y, sensitive, m_seed, n_sample, do_shuffle=True)

        # split into train, val and test
        x_train, x_val, x_test, sensitive_train, sensitive_val, sensitive_test, y_train, y_val, y_test = \
            split_data(x_sample, sensitive_sample, y_sample, s_attr, val_size=val_size, test_size=test_size)

        dataset_ext = DatasetExt(x_train=x_train, y_train=y_train, sensitive_train=sensitive_train,
                                 x_val=x_val, y_val=y_val, sensitive_val=sensitive_val,
                                 x_test=x_test, y_test=y_test, sensitive_test=sensitive_test,
                                 intercept=added_intercept)

        # construction of data folds
        dataset_folds = get_dataset_folds(dataset_ext, n_folds)

        # prune the tree
        assert fmccp or mccp
        start_time = time.time()

        trees, R_trees, alphas, n_terminals, p_rules, n_rules = get_cv_decision_trees(dataset_ext, dataset_folds,
                                                                                      tree_max, trees_max_folds, lmbda)

        runtime = time.time() - start_time

        # select the optimal pruned trees
        title_plot = method_name + ' ' + data_name
        print('\n acc cv')
        tree_acc = plot_cv_fmccp(trees, R_trees, None, alphas, n_terminals, dataset_ext, rep, title_plot, my_path)

        # write results
        write_results(tree_acc, rep, n_reps, runtime, cov, lmbda, p_epsilon, dataset_ext, R_trees, p_rules, n_rules, alphas,
                      containers, continous_idx_set)


    # save results
    save_results(containers,
                 column_names, continous_idx_set,
                 lmbda, cov, p_epsilon, n_reps, method_name, data_name, my_path,
                 prune_method=prune_method, constr_method=constr_method, experiment_variation=' acc cv')



