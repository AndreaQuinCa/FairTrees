# coding=utf-8
import matplotlib.pyplot as plt
import pydot
import numpy as np
import seaborn as sns

from tree_our_relab_prunning import make_node_ids
from evaluations_accuracy import get_split_accuracy, get_accuracy
from evaluations_fairness import get_node_fair_statistics, get_model_correlations, p_rules_model_stats
from data_utils import add_path

# %%


def plot_node(group, pred_idx_best, w_u_best, sensitive_attrs, cut_best, p_best, acc_best, depth):
    # unconstrained cut metrics
    p_best_u, n_best_u, cov_best_u = get_node_fair_statistics(group, pred_idx_best, w_u_best, sensitive_attrs, test=False)
    cut_best_u = -w_u_best[0] / w_u_best[1]
    print(cut_best, cut_best_u)
    acc_best_u = get_split_accuracy(pred_idx_best, w_u_best, group.x_test, group.y_test)
    x_i_train = group.x_train[:, pred_idx_best].reshape(-1, 1)
    # vals = len(np.unique(x_i_train))
    # if vals <= 2:
    #     x_i_train = x_i_train + np.random.uniform(-0.2, 0.2, x_i_train.shape)
    y_train = np.random.uniform(-1., 1., x_i_train.shape)
    x_y = np.hstack([np.zeros_like(x_i_train).reshape(-1, 1), x_i_train,
                     y_train])
    split_pred_cut_b, split_pred_cut_u = [1, cut_best], [1, cut_best_u]
    n_total = group.y_train.shape[0]
    n_non_protected = np.sum(group.sensitive_train[sensitive_attrs[0]])
    n_protected = n_total - n_non_protected
    perc = n_protected * 100 / n_non_protected
    plot_title = ' '.join(['Depth %d' % depth, 'Total %d' % n_total,
                           'Prot/non prot %0.2f %%' % perc])

    # Plot to decision lines
    plot_boundaries(split_pred_cut_u, split_pred_cut_b, p_best_u, p_best, acc_best_u, acc_best, plot_title,
                    x_y, group.y_train, group.sensitive_train, sensitive_attrs[0])


def plot_cv_mccp(trees, R_trees, alphas, n_terminals, dataset_ext, rep, data_name_folder, my_path):

    cp_plot = alphas / abs(R_trees[0])
    R_trees_plot = R_trees / abs(R_trees[0])

    x_tick_labels = [str(round(cp, 3)) + '\n n ter.  ' + str(n_ter) for cp, n_ter in zip(cp_plot, n_terminals)]
    x_tick_labels[0] = 'Inf'

    points_colors = []
    accs = []
    print('prunned trees n.terms and acc')
    for a, tree_alpha in enumerate(trees):
        accuracie = get_accuracy(tree_alpha, dataset_ext, test=True)
        points_colors.append(accuracie)
        accs.append((n_terminals[a], round(accuracie, 3)))
    points_colors = np.array(points_colors)
    tree_selection = np.argmin(R_trees_plot)
    label_x = r'$cp$'
    label_y = r'$Relative\quad R^{CV}(\alpha)$'
    title = data_name_folder + ' rep=%d' % rep
    path = add_path('img/' + 'prunning cv plots/', my_path)
    plot_xy(cp_plot, R_trees_plot, label_x, label_y, x_tick_labels, points_colors, tree_selection, title, path)

    print('selected tree ')
    print(accs[tree_selection])
    tree = trees[tree_selection]

    return tree


def plot_cv_fmccp(trees, R_trees, p_rules, alphas, n_terminals, dataset_ext, rep, data_name_folder, my_path):

    # x
    cp_plot = alphas / abs(R_trees[0])
    label_x = '# leafs'

    # color and size
    x_tick_labels = [str(n_ter) for n_ter in n_terminals]
    x_tick_labels[0] = '1'
    points_accs = []
    points_prules = []
    terms_accs_prule = []
    for a, tree_alpha in enumerate(trees):
        accuracie = get_accuracy(tree_alpha, dataset_ext, test=True)
        correlation_dict_test = get_model_correlations(tree_alpha, dataset_ext, test=True)
        s_attr_name = dataset_ext.sensitive_attrs[0]
        p_rule, _ = p_rules_model_stats(correlation_dict_test, s_attr_name, multiply=1.0)
        points_accs.append(accuracie)
        points_prules.append(p_rule)
        terms_accs_prule.append((n_terminals[a], round(accuracie, 3), p_rule))
    points_accs = np.array(points_accs)
    points_prules = np.array(points_prules)
    path = add_path('img/' + 'prunning cv plots/', my_path)

    # y and tree selection
    if p_rules is not None:

        mu, std = np.mean(R_trees), np.std(R_trees)
        R_trees_norm = np.array([(col - mu) / std for col in R_trees])

        R_fair = 1. - p_rules
        mu, std = np.mean(R_fair), np.std(R_fair)
        R_fair_norm = np.array([(col - mu) / std for col in R_fair])
        tree_selections = np.argwhere(np.diff(np.sign(R_trees_norm - R_fair_norm))).flatten()
        legends_y = [r'$R^{CV}(\alpha)', r'1-prule^{CV}(\alpha)$']
        title = data_name_folder + ' R and p-rule' + ' rep=%d' % rep
        plot_x_two_ys(cp_plot, R_trees_norm, R_fair_norm, label_x, legends_y, x_tick_labels, points_accs, points_prules, tree_selections, title,
                      path)

        tree_selection = tree_selections[-1]

    else:

        R_trees_plot = R_trees / abs(R_trees[0])
        label_y = r'$Relative\quad R^{CV}(\alpha)$'
        title = data_name_folder + ' R rep=%d' % rep
        tree_selection = np.argmin(R_trees_plot)
        plot_xy(cp_plot, R_trees_plot, label_x, label_y, x_tick_labels, points_accs, points_prules, tree_selection, title, path)

    print('tree max')
    print('terms_accs_prule', terms_accs_prule[-1])
    print('selected tree ')
    print('terms_accs_prule', terms_accs_prule[tree_selection])
    tree = trees[tree_selection]
    return tree


# for 2d graphics
def plot_xy(x, y, label_x, label_y, x_tick_labels, points_colors, points_sizes, special_point, title, path):
    cm = plt.cm.get_cmap('Wistia')

    m, M = points_colors.min(), points_colors.max()
    points_colors_norm = [(col-m)/(M-m) for col in points_colors]
    points_colors_norm = cm(points_colors_norm)

    m, M = points_sizes[1:].min(), points_sizes.max()
    points_sizes_norm = np.array([40 + (siz - m) / (M - m) * 200 for siz in points_sizes])

    plt.scatter(np.arange(len(x)), y, marker="o", color=points_colors_norm, s=points_sizes_norm)

    # tree selection
    px = np.arange(len(x))[special_point]
    py = y[special_point]
    plt.scatter(px, py, marker='d', color='blue', s=5)

    plt.xticks(np.arange(len(x)), x_tick_labels, rotation=70, fontsize=6)
    plt.subplots_adjust(bottom=0.15)

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    if path is not None:
        plt.savefig(path + "/" + title + ".png")
    plt.show()


def plot_x_two_ys(x, y1, y2, label_x, legends_y, x_tick_labels, points_colors_1, points_colors_2, special_points, title, path):
    x = np.arange(len(x))

    # R
    plt.plot(x, y1, '-', color='orange')
    # E
    plt.plot(x, y2, '-', color='blue')

    # info test
    cm1 = plt.cm.get_cmap('Oranges_r', 50)
    cm2 = plt.cm.get_cmap('Blues', 100)
    points_colors_1, points_colors_2 = points_colors_1[1:], points_colors_2[1:]
    m1, M = points_colors_1.min(), points_colors_1.max()
    points_colors_1 = [1-(col - m1) / (M - m1) for col in points_colors_1]
    points_colors_1 = cm1(points_colors_1)
    m2, M = points_colors_2.min(), points_colors_2.max()
    points_colors_2 = [(col - m2) / (M - m2) for col in points_colors_2]
    points_colors_2 = cm2(points_colors_2)
    plt.scatter(x[1:], y1[1:], marker="o", color=points_colors_1)
    plt.scatter(x[1:], y2[1:], marker="o", color=points_colors_2)

    # tree selection
    m1, m2 = min(y1), min(y2)
    m = min(m1, m2)
    plt.scatter(x[special_points], [m]*len(special_points), marker='d', color='green', s=20)

    # legends
    plt.xticks(np.arange(len(x)), x_tick_labels, rotation=70, fontsize=6)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(label_x)
    plt.title(title)
    if path is not None:
        plt.savefig(path + "/" + title + ".png")
    plt.legend(legends_y)
    plt.show()


# %%

# Plot scatter and lines

def grab_splits(node, depth=0, orientation='c', idx_father=0):
    splits = []
    if 'index' in node.keys() and 'cut' in node.keys():
        splits = [(node['index'], node['cut'], depth, orientation, idx_father)]
        if 'left' in node.keys() and 'right' in node.keys():
            splits_l = grab_splits(node['left'], depth + 1, 'l', node['index'])
            splits_r = grab_splits(node['right'], depth + 1, 'r', node['index'])
            splits = splits + splits_l + splits_r
    return splits


# TODO: Agregar p-value y cov por corte
def plot_splits(splits, acc, data_ext, fname):
    num_to_draw = 300  # we will only draw a small number of points to avoid clutter
    x_draw = data_ext.x_train[0:num_to_draw]
    y_draw = data_ext.y_train[0:num_to_draw]
    x_control_draw = data_ext.sensitive_train["s1"][:num_to_draw]

    X_s_0 = x_draw[x_control_draw == 0.0]
    X_s_1 = x_draw[x_control_draw == 1.0]
    y_s_0 = y_draw[x_control_draw == 0.0]
    y_s_1 = y_draw[x_control_draw == 1.0]
    plt.scatter(X_s_0[y_s_0 == 1.0][:, 1], X_s_0[y_s_0 == 1.0][:, 2], color='green', marker='x', s=30, linewidth=1.5)
    plt.scatter(X_s_0[y_s_0 == -1.0][:, 1], X_s_0[y_s_0 == -1.0][:, 2], color='red', marker='x', s=30, linewidth=1.5)
    plt.scatter(X_s_1[y_s_1 == 1.0][:, 1], X_s_1[y_s_1 == 1.0][:, 2], color='green', marker='o', facecolors='none',
                s=30)
    plt.scatter(X_s_1[y_s_1 == -1.0][:, 1], X_s_1[y_s_1 == -1.0][:, 2], color='red', marker='o', facecolors='none',
                s=30)

    cut_dim_1 = []
    cut_dim_2 = []

    for n_split, split in enumerate(splits):

        x1, x2 = min(x_draw[:, 1]), max(x_draw[:, 1])
        y1, y2 = min(x_draw[:, 2]), max(x_draw[:, 2])
        idx, cut, depth, orientation, idx_father = split[0], split[1], split[2], split[3], split[4]

        if idx == 1:
            x1, x2 = cut, cut
            cut_dim_1.append(cut)

            if len(cut_dim_2) > 0 and orientation == 'l':
                y2 = min(cut_dim_2)
            elif len(cut_dim_2) > 0 and orientation == 'r':
                y1 = max(cut_dim_1)
            else:
                if orientation == 'l': y2 = cut
                if orientation == 'r': y1 = cut

        elif idx == 2:
            y1, y2 = cut, cut
            cut_dim_2.append(cut)

            if len(cut_dim_1) > 0 and orientation == 'l':
                x2 = min(cut_dim_1)
            elif len(cut_dim_1) > 0 and orientation == 'r':
                x1 = max(cut_dim_1)
            else:
                if orientation == 'l': x2 = cut
                if orientation == 'r': x1 = cut

        else:
            assert (idx == 1 or idx == 2)

        plt.plot([x1, x2], [y1, y2], 'c-', linewidth=3, label="Acc=%0.2f" % acc)

    plt.savefig("img/" + fname + ".png")
    plt.show()


def plot_tree(tree, acc, data_ext, fname):
    """
    Plot a decision tree.

    :param tree: dict. The decision tree.
    :param acc: float. Its accuracy for label.
    :param data_ext: DatasetExt instance.
    :param fname:
    :return:
    """
    splits = grab_splits(tree)
    splits = sorted(splits, key=lambda tup: tup[2])
    plot_splits(splits, acc, data_ext, fname)


def plot_boundaries(split_pred_cut_1, split_pred_cut_2, p1, p2, acc1, acc2, title, x, y, x_sensitive, s_atrr):
    num_to_draw = int(x.shape[0]*.2) # we will only draw a small number of points to avoid clutter
    if num_to_draw < 100:
        num_to_draw = -1
    x_draw = x[:num_to_draw]
    y_draw = y[:num_to_draw]
    x_control_draw = x_sensitive[s_atrr][:num_to_draw]

    X_s_0 = x_draw[x_control_draw == 0.0]
    X_s_1 = x_draw[x_control_draw == 1.0]
    y_s_0 = y_draw[x_control_draw == 0.0]
    y_s_1 = y_draw[x_control_draw == 1.0]
    plt.scatter(X_s_0[y_s_0 == 1.0][:, 1], X_s_0[y_s_0 == 1.0][:, 2], color='green', marker='x', s=30, linewidth=1.5,
                label="Prot. +ve")
    plt.scatter(X_s_0[y_s_0 == -1.0][:, 1], X_s_0[y_s_0 == -1.0][:, 2], color='red', marker='x', s=30, linewidth=1.5,
                label="Prot. -ve")
    plt.scatter(X_s_1[y_s_1 == 1.0][:, 1], X_s_1[y_s_1 == 1.0][:, 2], color='green', marker='o', facecolors='none',
                s=30, label="Non-prot. +ve")
    plt.scatter(X_s_1[y_s_1 == -1.0][:, 1], X_s_1[y_s_1 == -1.0][:, 2], color='red', marker='o', facecolors='none',
                s=30, label="Non-prot. -ve")

    # plt.legend()
    x1, x2 = min(x_draw[:, 1]), max(x_draw[:, 1])
    y1, y2 = min(x_draw[:, 2]), max(x_draw[:, 2])

    if split_pred_cut_1[0] == 1:
        x1, x2 = split_pred_cut_1[1], split_pred_cut_1[1]
    else:
        y1, y2 = split_pred_cut_1[1], split_pred_cut_1[1]

    if type(p1) == str: p1 = -1.

    plt.plot([x1, x2], [y1, y2], 'c--', linewidth=3, label="Original: Acc=%0.0f; p-rule=%0.0f" % (acc1*100, p1*100))

    x1, x2 = min(x_draw[:, 1]), max(x_draw[:, 1])
    y1, y2 = min(x_draw[:, 2]), max(x_draw[:, 2])

    if split_pred_cut_2[0] == 1:
        x1, x2 = split_pred_cut_2[1], split_pred_cut_2[1]
    else:
        y1, y2 = split_pred_cut_2[1], split_pred_cut_2[1]

    if type(p2) == str: p2 = -1.

    plt.plot([x1, x2], [y1, y2], 'b-', linewidth=3, label="Constrained: Acc=%0.0f; p-rule=%0.0f" % (acc2*100, p2*100))
    plt.legend()
    # plt.savefig("img/" + fname + ".png")
    # plt.title(title)
    plt.show()


# %%

# Draw diagram of a decision tree

def get_legend(node, classes_names, sens_attr_name, n_training):
    id = str(node['id'])
    id += '\\n Depth: %2d' % node['depth']
    pred = node['pred']
    id += '\\n Pred: %2d' % pred

    if not node['is_terminal'] and 'p_value' in node.keys():
        id += '\\n Cov: %0.2f, P-value: %0.2f, N-Value %0.2f' % (node['mean_cov'], node['p_value'], node['n_value'])
    elif 'p_value' in node.keys():
        id += '\\n P-value: %0.2f, N-Value %0.2f' % (node['p_value'], node['n_value'])

    # class Y data
    acc = node['acc']*100.
    if pred == classes_names[0]:
        perc_0 = acc
        perc_1 = 100. - acc
    else:
        perc_0 = 100. - acc
        perc_1 = acc

    node_training_size = float(node['group'].training_size)
    perc_of_cases = node_training_size * 100. / n_training
    label_ls = [('Cases: %0.2f', perc_of_cases), ('%s class: %0.2f  %0.2f', ('%', perc_0, perc_1))]
    for (k, v) in label_ls:
        id += '\\n' + k % v

    # sensitive group A data
    sensitive_vals = node['group'].sensitive_train[sens_attr_name]
    n_total_non_protected = np.sum(sensitive_vals)
    n_total_protected = sensitive_vals.shape[0] - n_total_non_protected
    perc_0 = n_total_protected * 100 / node_training_size
    perc_1 = n_total_non_protected * 100 / node_training_size
    id += '\\n' + '%s group: %0.2f  %0.2f' % ('%', perc_0, perc_1)

    if not node['is_terminal']:
        id += '\\n Split: V%2d < %0.2f' % (node['index'], round(node['cut'], 2))

    return id


def draw_edge(graph, parent_name, child_name, pred, classes_to_idx):
    colors = ['blue', 'orange', 'yellow']
    edge = pydot.Edge(parent_name, child_name, color=colors[classes_to_idx[pred]])
    graph.add_edge(edge)


def graph_tree(graph, node, parent=None, classes_names=(-1, 1), sens_attr_name=None, n_training=None, classes_to_idx=None):
    # text in node
    node_legend = get_legend(node, classes_names, sens_attr_name, n_training)

    if parent:
        draw_edge(graph, parent, node_legend, int(node['pred']), classes_to_idx)

    # process child nodes
    keys = node.keys()
    if 'left' in keys or 'right' in keys:
        graph_tree(graph, node['left'], node_legend, classes_names, sens_attr_name, n_training, classes_to_idx)
        graph_tree(graph, node['right'], node_legend, classes_names, sens_attr_name, n_training, classes_to_idx)


def draw_tree(tree, filename):
    if 'id' not in tree:
        make_node_ids(tree)
    classes_names = tree['group'].classes_names
    classes_to_idx = tree['group'].classes_to_idx
    sens_attr_name = tree['group'].sensitive_attrs[0]
    n_training = tree['group'].training_size
    graph = pydot.Dot(graph_type='graph')

    graph_tree(graph=graph, node=tree, parent=None, classes_names=classes_names, sens_attr_name=sens_attr_name,
               n_training=n_training, classes_to_idx=classes_to_idx)

    graph.write('img/' + filename, prog=None, format='png')


