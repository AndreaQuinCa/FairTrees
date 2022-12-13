# encoding=utf-8
#%%
from collections import defaultdict
from visualization_trees import *
from evaluations_accuracy import prediction_set

# %%


def make_node_ids(node, node_number=1):
    node['id'] = node_number
    if not node['is_terminal']:
        number_of_nodes_left = make_node_ids(node['left'], node_number + 1)
        number_of_nodes_right = make_node_ids(node['right'], number_of_nodes_left + 1)
        return number_of_nodes_right
    return node_number


def get_terminal_ids(node, terminals_id):
    if node['depth'] == 0 and 'id' not in node.keys():
        make_node_ids(node)

    if node['is_terminal']:
        terminals_id.add(node['id'])
    else:
        get_terminal_ids(node['left'], terminals_id)
        get_terminal_ids(node['right'], terminals_id)


def get_cont_table(node, tree):

    # initialize contingency table
    contingency_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    sensitive_attrs = [0, 1]
    class_labels = [-1, 1]
    pred_labels = [-1, 1]
    for a in sensitive_attrs:
        for y in class_labels:
            for p in pred_labels:
                contingency_table[a][y][p] += 0

    # check variable format
    sens_attr_name = tree['group'].sensitive_attrs[0]
    predictors_vals, sensitive_vals, class_vals = node['group'].x_train, node['group'].sensitive_train[sens_attr_name],\
                                                  node['group'].y_train
    preds = prediction_set(node, predictors_vals)
    n_cases_node = node['group'].training_size
    assert (n_cases_node == predictors_vals.shape[0] == sensitive_vals.shape[0] == class_vals.shape[0] ==
            preds.shape[0])

    for (a, y, p) in zip(sensitive_vals, class_vals, preds):
        contingency_table[a][y][p] += 1

    # TODO: elegir como normalize: n_protected
    sensitive_vals = tree['group'].sensitive_train[sens_attr_name]
    n_total_non_protected = np.sum(sensitive_vals)
    n_total_protected = sensitive_vals.shape[0] - n_total_non_protected
    group_to_n_total = {0: float(n_total_protected), 1: float(n_total_non_protected)}
    # n_total_training = tree['group'].training_size

    for a in sensitive_attrs:
        for y in class_labels:
            for p in pred_labels:
                contingency_table[a][y][p] /= group_to_n_total[a]
                # contingency_table[a][y][p] /= (group_to_n_total[a] * n_total_training)  # así hice los experimentos

    return contingency_table


def calculate_terminals_cont_tables(node, tree, ids_to_cont_tables):
    if node['is_terminal']:
        ids_to_cont_tables[node['id']] = get_cont_table(node, tree)
    else:
        calculate_terminals_cont_tables(node['left'], tree, ids_to_cont_tables)
        calculate_terminals_cont_tables(node['right'], tree, ids_to_cont_tables)


def get_terminal_preds(node, ids_to_preds):
    if node['is_terminal']:
        ids_to_preds[node['id']] = node['pred']
    else:
        get_terminal_preds(node['left'], ids_to_preds)
        get_terminal_preds(node['right'], ids_to_preds)


def calculate_terminal_delta_discr(terminal_ids_to_c_tables, ids_to_pred):
    ids_to_d_discr = dict()
    for id, c_table in terminal_ids_to_c_tables.items():
        # contingency_table[a][y][p]
        protected_term = c_table[0][-1][1] + c_table[0][1][1]
        non_protected_term = c_table[1][-1][1] + c_table[1][1][1]
        discr = non_protected_term - protected_term

        if ids_to_pred[id] == -1:  # label would be changed to 1 so discrimination increases
            ids_to_d_discr[id] = discr
        else:  # label would be changed to -1, discrimination decreases
            ids_to_d_discr[id] = -discr
    return ids_to_d_discr


def calculate_terminal_delta_acc(candidates, terminal_ids_to_c_tables, ids_to_pred):
    ids_to_d_acc = dict()

    for id, c_table in terminal_ids_to_c_tables.items():
        if id not in candidates:
            continue
        # contingency_table[a][y][p]
        n_cases_negative = c_table[0][-1][-1] + c_table[1][-1][-1] + c_table[0][-1][1] + c_table[1][-1][1]
        n_cases_positive = c_table[0][1][1] + c_table[1][1][1] + c_table[0][1][-1] + c_table[1][1][-1]

        if ids_to_pred[id] == 1:
            ids_to_d_acc[id] = n_cases_negative - n_cases_positive
        else:
            ids_to_d_acc[id] = n_cases_positive - n_cases_negative

    return ids_to_d_acc


def calculate_tree_discr(tree):
    c_p_table = get_cont_table(tree, tree)
    non_protected_term = c_p_table[1][-1][1] + c_p_table[1][1][1]
    protected_term = c_p_table[0][-1][1] + c_p_table[0][1][1]
    return non_protected_term - protected_term


def calculate_discr(terminals_to_relabel, terminal_to_d_discr, tree_discr):
    if not terminals_to_relabel:
        return tree_discr
    d_discr_vals = [d_discr for id, d_discr in terminal_to_d_discr.items() if id in terminals_to_relabel]
    return tree_discr + np.sum(d_discr_vals)


def get_best_terminal_to_relab(candidates, id_to_d_acc, id_to_d_discr):
    # id_to_d_acc = {id: d_acc for id, d_acc in id_to_d_acc.items()}
    id_to_cocients = [(id, id_to_d_discr[id] / float(id_to_d_acc[id])) if id_to_d_acc[id] != 0 else (id, np.inf) for id in candidates]
    opt_idx = max(range(len(id_to_cocients)), key=lambda i: id_to_cocients[i][1])
    opt_id = id_to_cocients[opt_idx][0]
    return opt_id


def select_terminal_to_relabel(candidates, candidates_to_d_acc, terminal_to_d_discr, tree_discr, epsilon):
    terminals_to_relabel = set()
    while calculate_discr(terminals_to_relabel, terminal_to_d_discr, tree_discr) > epsilon and candidates:
        best_term_to_relab = get_best_terminal_to_relab(candidates, candidates_to_d_acc, terminal_to_d_discr)
        terminals_to_relabel.add(best_term_to_relab)
        candidates.remove(best_term_to_relab)
    return terminals_to_relabel


def relabel(term_ids_to_relabel, node):
    if not term_ids_to_relabel:
        return
    if node['id'] in term_ids_to_relabel:
        assert (node['is_terminal'])
        node['pred'] = - node['pred']
    elif not node['is_terminal']:
        relabel(term_ids_to_relabel, node['left'])
        relabel(term_ids_to_relabel, node['right'])


def fair_relabel(tree, p_epsilon):
    tree_discr = calculate_tree_discr(tree)
    epsilon = p_epsilon * tree_discr
    make_node_ids(tree)
    terminal_ids_to_c_tables = dict()
    calculate_terminals_cont_tables(tree, tree, terminal_ids_to_c_tables)  # terminal to contingency table
    ids_to_preds = dict()
    get_terminal_preds(tree, ids_to_preds)
    ids_to_d_discr = calculate_terminal_delta_discr(terminal_ids_to_c_tables, ids_to_preds)  # terminal to discrimination delta dict
    candidates = set(id for id, d_discr in ids_to_d_discr.items() if d_discr < 0.)  # terminal id set
    candidates_to_d_acc = calculate_terminal_delta_acc(candidates, terminal_ids_to_c_tables, ids_to_preds)  # terminal to accuracy delta dict
    terminals_to_relabel = select_terminal_to_relabel(candidates, candidates_to_d_acc, ids_to_d_discr, tree_discr, epsilon)
    relabel(terminals_to_relabel, tree)
    tree_discr_final = calculate_tree_discr(tree)
    return tree_discr, tree_discr_final, terminals_to_relabel

