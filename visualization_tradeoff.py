import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns

# %%
# parameters
hue = 'lambda'
color_fact = 1
data_name = 'law'
method = 'FMCCP'
type_of_table = 'metrics'
just_trees = True
runtime = False

violin = True
boxplot = True
swarmplot = True
density = False
histogram = False

#%%

if hue == 'Method':
    if just_trees:
        experiment = 'CF CD CI CA TLR'
        hue_order = ['TLR', 'CF-TLR', 'CD-TLR', 'CI-TLR']
        palette = sns.color_palette("Set2")
    else:
        experiment = 'TLR CF-TLR LR C-LR'
        hue_order = ['TLR', 'LR', 'CF-TLR', 'C-LR']
        palette = sns.color_palette(['deepskyblue', 'royalblue', 'yellowgreen', 'forestgreen'])
        hues_palettes = zip([['CF-TLR', 'C-LR'], ['TLR', 'LR']],
                            [sns.color_palette(['yellowgreen', 'forestgreen']),
                             sns.color_palette(['deepskyblue', 'royalblue'])])
elif hue == 'lambda':
    experiment = 'Gini Gini+FMCCP'
    hue_order = None
    palette = sns.color_palette("rocket")
    unc_method = 'Gini'
    cons_method = 'Gini+FMCCP'
    cmap = sns.cubehelix_palette(start=1.2, as_cmap=True, reverse=False, light=.6, gamma=1.2, rot=.5)
    bar_label = ' constrain'

elif hue == 'cov':
    if method == 'LR':
        experiment = 'LR_C-LR'
        unc_method = 'LR'
        cons_method = 'C-LR'
    else:
        experiment = 'LRT_C-LRT'
        unc_method = 'LRT'
        cons_method = 'C-LRT'

    hue_order = None
    palette = sns.color_palette("rocket")
    cmap = sns.cubehelix_palette(start=2., as_cmap=True, reverse=True, light=.6, gamma=1.2, rot=.7)
    bar_label = ' normalized constrain'

elif hue == 'p-epsilon':
    experiment = 'Gini Gini+RELAB'
    hue_order = None
    unc_method = 'Gini'
    cons_method = 'Gini+RELAB'
    # palette = ['goldenrod', 'mediumseagreen', 'dodgerblue'] + sns.color_palette("rocket")[-4:]
    cmap = sns.cubehelix_palette(start=2., as_cmap=True, reverse=True, light=.6, gamma=1.2, rot=.7)
    bar_label = ' epsilon'

#%%
# data loading
folder_specific_experiment = "to_compare_" + experiment
folder_experiments = "/".join(['tables', folder_specific_experiment, data_name, type_of_table])
folder_experiments_img = "/".join([folder_experiments, 'img'])
folder_experiments_csv = "/".join([folder_experiments, '*.csv'])
filenames = glob.glob(folder_experiments_csv)

n_methods = len(filenames)
method_results = [0] * n_methods
for i, filename in enumerate(filenames):
    df = pd.read_csv(filename, sep=',')
    assert df.shape[0] == 30
    method_results[i] = df

#%%
# format dataframes
if type_of_table == 'metrics':
    plot_cols = [u'Accuracy', u'p-rule',
                 u'n-rule', u'Method', u'Set']
    if hue == 'cov':
        plot_cols = [u'Method', u'cov', u'Accuracy', u'p-rule',
                     u'n-rule', u'SP', u'constant']
    if hue == 'lambda':
        plot_cols = [u'Method', u'lambda', u'Accuracy', u'p-rule',
                     u'n-rule', 'SP', u'constant']
    if hue == 'p-epsilon':
        plot_cols = [u'Method', u'p-epsilon', u'Accuracy', u'p-rule',
                     u'n-rule', 'SP', u'constant']
    if runtime:
        plot_cols = [u'Runtime', u'Method', u'Set']
    sns.set_style('darkgrid', {'axes.facecolor': '.9'})
if type_of_table == 'metrics by depth':
    plot_cols = list(method_results[0].columns)

val_to_nconstant = dict()
for idx, results in enumerate(method_results):
    results = results[plot_cols]

    if results['Method'][0] in ['LRT', 'Gini', 'LR']:
        results[hue] = np.zeros(results.shape[0]) - 1
    else:
        print('lambda', results.loc[:, hue][0])
        results.loc[:, hue] = 1 - results[hue]
        print('1-lambda', results.loc[:, hue][0])


    results_hue_val = set(list(results[hue]))
    assert(len(results_hue_val) == 1)
    val_to_nconstant[results_hue_val.pop()] = np.sum(results['constant'])
    method_results[idx] = results


#%%
from matplotlib import cm
# cmap = sns.cubehelix_palette(as_cmap=True, reverse=True)
# cmap = sns.cubehelix_palette(start=4., rot=.4, as_cmap=True, reverse=True)
# cmap = cm.get_cmap('cool', 10)
# method_color = {'TLR': 'g', 'LR': 'm'}
# cmap = plt.cm.get_cmap('Greens')

#

def make_means_df(method_results):
    df_means_list = []
    for df in method_results:
        means = df.mean()
        method = set(list(df['Method']))
        assert (len(method) == 1)
        means['Method'] = method.pop()
        df_means_list.append(means)

    df_plot = pd.DataFrame(df_means_list)

    df_plot.sort_values(by=[hue], inplace=True)
    df = df_plot.reset_index(drop=True)

    df._set_value(0, hue, np.nan)

    cols = list(df.columns)
    o_cols = ['Method'] + cols[:-1]
    df = df[o_cols]
    m_cols = ['Method'] + [hue] + ['Mean ' + col for col in cols[1:-1]]
    df.columns = m_cols
    m_cols.remove('Mean constant')
    df = df[m_cols]
    return df_plot, df.round(5)


def make_vars_df(method_results):
    df_list = []
    for df in method_results:

        vars = df.var()

        method = set(list(df['Method']))
        assert (len(method) == 1)
        vars['Method'] = method.pop()

        hues = set(list(df[hue]))
        assert (len(hues) == 1)
        vars[hue] = hues.pop()

        df_list.append(vars)

    df = pd.DataFrame(df_list)

    df.sort_values(by=[hue], inplace=True)
    df = df.reset_index(drop=True)

    df._set_value(0, hue, np.nan)

    cols = list(df.columns)
    o_cols = cols[1:-1]
    df = df[o_cols]
    m_cols = [col + ' var' for col in cols[1:-1]]
    df.columns = m_cols
    m_cols.remove('constant var')
    df = df[m_cols]

    return df.round(5)


def plot_rule_vs_acc_means(df):
    rule_names = ["p-rule", "n-rule", "SP"]

    y_name = "Accuracy"
    df_unconstrained = df[df['Method'] == unc_method]
    df = df[df['Method'] != unc_method]
    for x_name in rule_names:
        fig, ax = plt.subplots()
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        rule, accuracy = df[x_name].to_numpy(), df['Accuracy'].to_numpy()
        constraints = df[hue].to_numpy()

        m, M = constraints.min(), constraints.max()
        points_colors = np.array([((col - m) / (M - m)) for col in constraints])
        points_colors = np.power(points_colors, (1/color_fact))
        points = ax.scatter(rule, accuracy, cmap=cmap, c=points_colors, alpha=0.8)
        fig.colorbar(points, ax=ax, label=cons_method + bar_label, ticks=None)

        rule, accuracy = df_unconstrained[x_name].to_numpy(), df_unconstrained['Accuracy'].to_numpy()
        ax.scatter(rule, accuracy, c='b', label=unc_method, marker='o', alpha=0.4)
        plt.legend()
        plot_name = 'trade off of: ' + ' '.join([cons_method, data_name, x_name])
        plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
        plt.show()


means_plot, means_df = make_means_df(method_results)
vars_df = make_vars_df(method_results)


#%%
# plot_rule_vs_acc_means(means_plot)


#%%
filename = ''.join([folder_experiments, '/', 'tables/', data_name, '_means_vars', '.csv'])
df = pd.concat([means_df, vars_df], axis=1)
df.to_csv(filename, index=False)

#%%

