# coding=utf-8
# %%
from evaluations_fairness import *
from data_utils import DatasetExt

# %%

def count_nodes(node, node_number=1):
    if node['is_terminal']:
        return node_number
    else:
        number_of_nodes_left = count_nodes(node['left'], node_number + 1)
        number_of_nodes_right = count_nodes(node['right'], number_of_nodes_left + 1)
        return number_of_nodes_left + number_of_nodes_right


def calculate_terminals_of_node(node):
    """
    Calculate number of terminals of each of the descendants of the node
    in the first call.

    ---------
    Parameters

    param node: dict. A branch.
    return: None. The number of terminals of each descendant are saved in themselfs.
    """
    if node['is_terminal']:
        node['n_terminals'] = 1
    else:
        calculate_terminals_of_node(node['left'])
        calculate_terminals_of_node(node['right'])
        node['n_terminals'] = node['left']['n_terminals'] + node['right']['n_terminals']
    pass


#%%

# Node misclassification estimate

# Estimate conjugate probabilities: p(k,t)
def estimate_conj_prob_class_node(group):
    # count instances of classes in node: N_k(t)

    sizes_class_node = np.zeros_like(group.aprioris)
    node_classes = group.y_train
    classes_to_idx = group.classes_to_idx
    for class_val, idx_class in classes_to_idx.items():
        class_val_arr = np.zeros_like(node_classes) + class_val
        size_class_node = np.sum(node_classes == class_val_arr) * 1.0
        sizes_class_node[idx_class] += size_class_node

    # calculate proportion N_k(t)/N_k. N_k=total of class k in training
    prob_class = np.divide(sizes_class_node, group.classes_sizes)

    # estimate conjugate probabilities: p(k,t)
    return group.aprioris * prob_class


# Estimate ponderate misclassification per node: R(t)
def estimate_mis_node(group):
    """
    R(t) = p(t)r(t)

    r(t) = sum_{k neq k(t)} c(i | k) p(k | t),

    with k(t)= prediction of terminal t

    :param group:
    :return:
    """

    # p(k, t)
    conj_prob = estimate_conj_prob_class_node(group)

    # estimate probability a case falls into node p(t)
    p_node = conj_prob.sum()

    # estimate conditional probabilities p(k|t)
    cond_probs = conj_prob / p_node
    class_idx = np.argmax(cond_probs)
    pred = group.idx_to_class[class_idx]

    # find minimal misclassification: r(t)
    prods = np.matmul(group.costs, cond_probs)
    r = prods.min()

    # ponderate misclassification: R(t), prediction
    R = r * p_node

    return R, pred


# %%

def node_to_terminal(node):
    """
    Setting a branch to terminal, auxiliary for constructing and prunning.
    -----
    Parameters

    param node: dict. Node of a decision tree.
    return: dict. Node with format of terminal.
    """
    if 'cut' in node.keys():
        del (node['cut'])
        del (node['index'])
    if 'left' in node.keys():
        del (node['left'])
        del (node['right'])
    node['R_branch'] = node['R_node']
    node['n_terminals'] = 1
    node['is_terminal'] = True
    pass


def group_to_terminal(group, R=None, pred=None, depth=0):
    """
    Setting a branch to terminal, auxiliary for construction and prunning.

    -----
    Parameters

    param group: DatasetExt instance. Information of data in the group (left or right) of a split node.
    param R: float. Missclasification estimate of the node.
    return: dict. Terminal format.

    """
    assert group.sensitive_train

    if R is None:
        R, pred = estimate_mis_node(group)

    acc = get_group_acc(group, pred, test=False)
    p, n = get_terminal_fair_statistics(group, pred, test=False)
    R_fair = 1. - estimate_rule_node(group)

    return {'n_terminals': 1, 'is_terminal': True, 'group': group,
            'R_node': R,  'R_fair': R_fair, 'p_value': p, 'n_value': n,
            'pred': pred, 'acc': acc, 'depth': depth}


# %%

# Tree construction
def split_cases_by_cut(index, cut, group):
    """
    Split a dataset based on a predictor and a value.

    -----
    Parameters

    param index: int. Column in the trainig data associated to the predictor in optimal split.
    param cut: float. Cut in optimal split.
    param group: DatasetExt instance. Group of cases to be split.
    return: Two DatasetExt instances. Left and right groups detemined by the split.
    """

    idx_left = group.x_train[:, index] < cut
    idx_right = idx_left == 0

    left_x_tr, right_x_tr = group.x_train[idx_left], group.x_train[idx_right]
    left_y_tr, right_y_tr = group.y_train[idx_left], group.y_train[idx_right]

    left_sensitive_tr, right_sensitive_tr = None, None
    if group.sensitive_train:
        for attr in group.sensitive_attrs:
            left_sensitive_tr, right_sensitive_tr = {attr: group.sensitive_train[attr][idx_left]}, {
                attr: group.sensitive_train[attr][idx_right]}
    # TODO: revisar si es necesario paser info del padre: aprioris y costs
    left = DatasetExt(x_train=left_x_tr, y_train=left_y_tr, x_test=left_x_tr, y_test=left_y_tr,
                      sensitive_train=left_sensitive_tr,
                      sensitive_test=left_sensitive_tr, intercept=group.intercept,
                      classes_names=group.classes_names, classes_sizes=group.classes_sizes,
                      costs=group.costs, aprioris=group.aprioris, sensitive_attrs=group.sensitive_attrs,
                      sensitive_aprioris= group.sensitive_aprioris, sensitive_names=group.sensitive_names,
                      sensitive_sizes=group.sensitive_sizes)

    right = DatasetExt(x_train=right_x_tr, y_train=right_y_tr, x_test=right_x_tr, y_test=right_y_tr,
                       sensitive_train=right_sensitive_tr,
                       sensitive_test=right_sensitive_tr, intercept=group.intercept,
                       classes_names=group.classes_names, classes_sizes=group.classes_sizes,
                       costs=group.costs, aprioris=group.aprioris, sensitive_attrs=group.sensitive_attrs,
                       sensitive_aprioris=group.sensitive_aprioris, sensitive_names=group.sensitive_names,
                       sensitive_sizes=group.sensitive_sizes)
    return left, right


def split(node, max_depth, min_size, depth, get_split, *args):
    """
    Recursive construction of a decision tree.

    ----------
    Parameters

    param node: dict. A branch or a tree.
    param max_depth: int.
    param min_size: int
    param depth: int. Depht of current node in the tree of the first call.
    param get_split: function. It receives a dataset (instance of DatasetExt) and returns the optimal split.
     A split is determined by a predictor and a float number called cut.
    return: None. A decision tree is saved in place of node.

    ----------
    Notes

    The first call is of the form split(tree, max_depth, min_size, 0, get_split), where tree
    is a tree decision of depht one.

    As get_split functions, currently, we had implemented in tree_construction_gini.py the usual optimal split used in
    CART, with gini function. And in tree_construction_fair.py we had implemented an alternative method to find optimal
    cut using regression with fairness constraints in each predictor. The optimal split returned is the one that
    maximizes accuracy.

    """
    left_group, right_group = node['groups']
    del(node['groups'])

    # check ending criteria: maximum depht, purity and non-split
    if depth >= max_depth:
        node_to_terminal(node)
        return
    if left_group.training_size == 0 or right_group.training_size == 0:
        node_to_terminal(node)
        return
    clsses = np.unique(node['group'].y_train, return_counts=False)
    if clsses.shape[0] == 1:
        node_to_terminal(node)
        return
    node['is_terminal'] = False

    # left child
    if left_group.training_size <= min_size:
        node['left'] = group_to_terminal(left_group, depth=depth+1)
    else:
        node['left'] = get_split(left_group, depth+1, *args)
        split(node['left'], max_depth, min_size, depth + 1, get_split, *args)

    # right child
    if right_group.training_size <= min_size:
        node['right'] = group_to_terminal(right_group, depth=depth+1)
    else:
        node['right'] = get_split(right_group, depth+1, *args)
        split(node['right'], max_depth, min_size, depth + 1, get_split, *args)


def build_tree(max_depth, min_size, dataset_ext, get_split, *args):
    """
    Build a decision tree.

    ----------
    Parameters

    param max_depth: int. Maximum height allowed.
    param min_size: int. Minimum size of training cases in a node.
    param dataset_ext: DatasetExt instance. A class with training, test data. And info necessary to grow the tree.
    (See data_utils.py)
    param get_split: function. It receives a dataset (instance of DatasetExt) and
        returns the optimal split. A split is determined by a predictor and a float number called cut.
    param *args. Optional arguments of get_split function.
    return: dict. A decision tree.
    
    ----------
    Notes

    As get_split functions, currently, we had implemented in tree_construction_gini.py the usual optimal split used in
    CART, with gini function. And in tree_construction_fair.py we had implemented an alternative method to find optimal
    cut using regression with fairness constraints in each predictor. The optimal split returned is the one that
    maximizes accuracy.

    """
    root = get_split(dataset_ext, 0, *args)
    idx, cut, mean_cov = root['index'], root['cut'], root['mean_cov']  # save in case tree has depth=1
    split(root, max_depth, min_size, 0, get_split, *args)

    if root['is_terminal']:  # case tree has depth=1
        root['index'], root['cut'], root['mean_cov'] = idx, cut, mean_cov

    return root


