# coding=utf-8
import copy
from tree_construction import *


# %%

def get_tree_values_list(tree, key, vals_list):
    vals_list.append(tree[key])
    if not tree['is_terminal']:
        get_tree_values_list(tree['left'], key, vals_list)
        get_tree_values_list(tree['right'], key, vals_list)
    pass


def normalize_tree_values(tree, key, max_val):
    tree[key] = tree[key]/max_val
    if not tree['is_terminal']:
        normalize_tree_values(tree['left'], key, max_val)
        normalize_tree_values(tree['right'], key, max_val)
    pass


def normalize_error_and_unfairness(tree):
    r_error_list, r_unfair_list = [], []
    get_tree_values_list(tree, 'R_node', r_error_list)
    get_tree_values_list(tree, 'R_fair', r_unfair_list)
    max_error, max_unfair = max(r_error_list), max(r_unfair_list)
    normalize_tree_values(tree, key='R_node', max_val=float(max_error))
    normalize_tree_values(tree, key='R_fair', max_val=float(max_unfair))


def add_node_fair_regularization(node, lmbda):

    node['R_node'] = (1. - lmbda) * node['R_node'] + lmbda * node['R_fair']

    if not node['is_terminal']:
        add_node_fair_regularization(node['left'], lmbda)
        add_node_fair_regularization(node['right'], lmbda)


# %%

def estimate_mis_branch(node):
    """
    Estimate misclassification R(T_t) of the branch rooted at each node t
    of branch of first call.

    ---------
    Parameters

    param node: dict. A branch.
    return: None. The R(T_t) of each branch are saved in the root of the branch.

    --------
    Notes

        T_t branch of tree T, rooted at node t.
        R(T_t) = sum_{l terminal of T_t} R(l)

    """
    if node['is_terminal']:
        node['R_branch'] = node['R_node']
    else:
        estimate_mis_branch(node['left'])
        estimate_mis_branch(node['right'])
        node['R_branch'] = node['left']['R_branch'] + node['right']['R_branch']
    pass


def linkage_function(node):
    """
    Calculate the measure of the linkage g(t) of the branch rooted in the node t.
    g(t) = (misclassification estimate node t - misclassification estimate branch T_t)/(n terminals T_t -1)

    ---------
    Parameters

    param node: dict. A branch.
    return: g(t)
    """
    if node['is_terminal']:
        return np.inf

    return (node['R_node'] - node['R_branch']) / (node['n_terminals'] - 1)


def calculate_linkages(tree, linkages):
    """
    Calculate all linkage measures in a tree.

    ---------
    Parameters

    param tree: dict. A decision tree.
    param linkages: list. Should be an empty list in the first call.
    return: None. The measures are saved in linkages arr and the corresponding
    measure is also saved in it's associated node.
    """
    if not tree['is_terminal']:
        calculate_linkages(tree['left'], linkages)
        calculate_linkages(tree['right'], linkages)

    link = linkage_function(tree)
    tree['link'] = link
    linkages.append(link)
    pass


def find_weakest_link(tree):
    """
    Find the weakest linkage.

    ---------
    Parameters

    param tree: dict. A decision tree.
    return: float. The value of the weakest linkage in the tree: argmin g(t) (for each node t in the tree).

    """

    # values needed for calculate the linkage
    calculate_terminals_of_node(tree)
    estimate_mis_branch(tree)

    # get linkages of all the nodes
    linkages_list = []
    calculate_linkages(tree, linkages_list)
    weakest_link = min(linkages_list)
    return weakest_link


def prun_weakest(tree, weakest_link, epsilon=10e-6):
    """
    All descendants of the weakest link(s) are erased.

    ---------
    Parameters

    param tree: dict. Tree (branch/node) to be prunned.
    return: dict. Tree with the weakest nodes prunned.
    """
    T = copy.deepcopy(tree)
    T = prun_weakest_in_place(T, weakest_link, epsilon=epsilon)
    return T


def prun_weakest_in_place(tree, weakest_link, epsilon=10e-6):
    """
    All descendants of the weakest link(s) are erased.

    ---------
    Parameters

    param tree: dict. Tree (branch/node) to be prunned.
    return: dict. Tree with the weakest nodes prunned.
    """
    if abs(tree['link'] - weakest_link) < epsilon:
        node_to_terminal(tree)
    elif not tree['is_terminal']:
        tree['left'] = prun_weakest(tree['left'], weakest_link)
        tree['right'] = prun_weakest(tree['right'], weakest_link)
    return tree


def minimal_complexity_prunning(tree, lmbda=None, epsilon=10e-4):
    """
    Find {alpha_i, T(alpha_i)} sequence of the tree max tree with
    optimal complex prunning.

    ---------
    Parameters

    param tree_max: dict.
    return: a pair of lists [alphas, trees]:
    - trees: sequence of optimal prunning subtrees {T_k}
    - alphas: sequence of complexity parameter alpha_{k+1} = g(t_k)
    starting in 0 and ending in alpha associated with root subtree.

    """

    if lmbda is not None:  # fmccp
        normalize_error_and_unfairness(tree)
        add_node_fair_regularization(tree, lmbda)
        alpha = -np.inf
    else:
        alpha = 0.

    fail_increment_alpha = list()
    alphas = [alpha]
    trees = [tree]

    # calculate_terminals_of_node(tree)
    while not tree['is_terminal']:
        # prunning
        alpha_next = find_weakest_link(tree)
        tree = prun_weakest(tree, alpha_next)

        # should be an increasing sequence
        if alpha - alpha_next > epsilon:
            fail_increment_alpha.append((alpha_next, alpha))
            print('alpha increment failed')

        # save
        alphas.append(alpha_next)
        trees.append(tree)

        # actualization
        alpha = alpha_next

    if len(fail_increment_alpha) > 0:
        print('Fail increment')
        print(fail_increment_alpha)
        raise Exception('Fail increment')

    return np.array(alphas), trees


# %%

def select_elbow_index(rel_R_alpha, frac=15.0):
    R_min, R_max = min(rel_R_alpha), max(rel_R_alpha)
    radio = abs(R_max - R_min) / frac
    idx_best = np.argmin(rel_R_alpha)
    if len(rel_R_alpha) > 1:
        for idx in range(len(rel_R_alpha) - 1):
            if rel_R_alpha[idx + 1] < R_min + radio:
                idx_best = idx + 1
                break
        return idx_best
    else:
        return 0
