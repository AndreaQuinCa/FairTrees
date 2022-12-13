# coding=utf-8
# %%
import time

from data_utils import DatasetExt
from data_loading_new import *
from data_saving import *
from tree_construction import build_tree
from tree_construction_fair import get_fair_split
from tree_construction_gini import get_gini_split
from tree_loss_functions import logistic_loss

# %%

# Parameters

# experiments
n_reps = 30  # I
SEED = 110979
seed(SEED)
np.random.seed(SEED)

# data
data_name = 'law'  # I
continuous = False  # I
if continuous:
    data_name = data_name + ' continuous'
filename_parameters = 'parameters_experiments_' + data_name + '_c-lrt.csv'  # I
my_path = None  # I
val_size = 0.2  # I
test_size = 0.2  # I


# tree
get_split = get_fair_split  # I
max_depth = 10  # I
min_size = 5
draw = False

# fairness
loss_function = logistic_loss
apply_accuracy_constraint = 0
sep_constraint = 0

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
column_names = get_report_col_names(continous_idx_set, max_depth, fmccp=False)

# %%

# make experiments
for idx, parameter in parameters_experiments.iterrows():

    # get experiment paremeters
    method_name, apply_fairness_constraints, sensitive_attrs_to_cov_thresh, cov, mccp, fmccp, relab, lmbda, p_epsilon \
        = format_parameters(parameter, s_attr)
    constr_method = parameter['constr_method']
    if 'LRT' in constr_method:
        assert get_split == get_fair_split
    else:
        assert 'Gini' in constr_method
        assert get_split == get_gini_split
    assert parameter['prune_method'] == 'None'

    # for saving results
    containers = get_result_containers(n_reps, column_names)

    # construct Decisions Trees in n_reps samples
    for rep in range(n_reps):
        print(" N. repetition = %d " % rep)

        # get sample
        rep_seed = SEED * (rep + 1)
        x_sample, y_sample, sensitive_sample = get_sample(x, y, sensitive, rep_seed, n_sample, do_shuffle=True)

        # split into train, val and test
        x_train, x_val, x_test, sensitive_train, sensitive_val, sensitive_test, y_train, y_val, y_test = \
            split_data(x_sample, sensitive_sample, y_sample, s_attr, val_size=val_size, test_size=test_size)

        # to manage data
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
                              sensitive_attrs_to_cov_thresh, lmbda)
        runtime = time.time() - start_time

        # write results
        write_results(tree, rep, n_reps, runtime, cov, lmbda, p_epsilon, dataset_ext, None, None, None, None,
                      containers, continous_idx_set)

        if draw:
            save_tree(tree, p_epsilon, rep, method_name, data_name)

    # save results
    save_results(containers, column_names, continous_idx_set,
                lmbda, cov, p_epsilon, n_reps, method_name, data_name, my_path,
                prune_method=None, constr_method=constr_method, experiment_variation=None)


