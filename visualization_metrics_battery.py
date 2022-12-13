# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '.9'})


# %%

# datasets = ['adult', 'compas', 'ricci', 'law', 'adult continuous', 'compas continuous', 'ricci continuous',
#             'law continuous']

data_name = 'adult'
reps = 30
legend_pos = 0.16
# parameters
hue = 'lambda'
report_constant = True
method = 'FMCCP'

type_of_table = 'metrics'
# type_of_table = 'metrics by depth'
just_trees = True
runtime = False

violin = False
boxplot = True
swarmplot = False

density = False
histogram = False
qqplot = False

if hue == 'Method':
    if just_trees:
        experiment = 'LRT_C-LRT'
        # experiment = 'Gini Gini+FMCCP'
        # experiment = 'C-TLR C-TLR+MCCP C-TLR+FMCPP C-TLR+RELAB'
        # hue_order = None
        hue_order = ['LRT', 'C-LRT']
        # palette = sns.color_palette("rocket")
        # palette.reverse()
        palette = sns.color_palette(['deepskyblue', 'royalblue'])
        # palette = ['black'] + palette
        # palette = sns.color_palette("Set2")[:len(hue_order)]
        # palette = [sns.color_palette("Paired")[-3], sns.color_palette("Paired")[1], sns.color_palette("Paired")[-1]]
        # palette = [sns.color_palette("mako")[4], sns.color_palette("Paired")[-3]]
        # palette = [sns.color_palette("Paired")[6], sns.color_palette("Paired")[7],
        #            sns.color_palette("mako")[4], sns.color_palette("Paired")[-3]]
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
    # palette = sns.color_palette("rocket", as_cmap=True)

elif hue == 'cov':
    if method == 'LR':
        experiment = 'LR_C-LR'
    else:
        experiment = 'LRT_C-LRT'
    hue_order = None
    palette = sns.color_palette("rocket")
    palette.reverse()
    palette = ['black'] + palette

elif hue == 'p-epsilon':
    experiment = 'Gini Gini+RELAB'
    # experiment = 'CF-CARTLR CF-TLR CF-FCARTLR CF-TLR-RELAB-MACC'
    hue_order = None
    # palette = ['goldenrod', 'mediumseagreen', 'dodgerblue'] + sns.color_palette("rocket")[-4:]
    # palette = ['mediumseagreen'] + sns.color_palette("rocket")[-4:]


# get filenames
folder_specific_experiment = "to_compare_" + experiment
folder_experiments = "/".join(['tables', folder_specific_experiment, data_name, type_of_table])
folder_experiments_img = "/".join([folder_experiments, 'img'])
folder_experiments_csv = "/".join([folder_experiments, '*.csv'])
filenames = glob.glob(folder_experiments_csv)
n_methods = len(filenames)
palette = sns.color_palette("rocket", n_colors=n_methods)

# %%

# load csv's as dataframes
n_methods = len(filenames)
method_results = [0] * n_methods
for i, filename in enumerate(filenames):
    results = pd.read_csv(filename, sep=',')
    assert results.shape[0] == reps
    if results['Method'][0] in ['Gini', 'LRT', 'LR']:
        results.loc[:, hue] = np.zeros(results.shape[0]) - 1
    method_results[i] = results

# check format of dataframes
if type_of_table == 'metrics':
    plot_cols = [u'Accuracy', u'p-rule', u'n-rule', u'SP']
    plot_cols = plot_cols + [hue]
elif type_of_table == 'terminals':
    plot_cols = [u'n terminals']
    plot_cols = plot_cols + [hue]
elif type_of_table == 'metrics by depth':
    cols = method_results[0].columns
    new_cols = [col.replace("value", "rule") for col in cols]
    new_cols_dict = {col: col.replace("value", "rule") for col in cols}
    plot_cols = list(new_cols)[:-1]
elif runtime:
    plot_cols = [u'Runtime', u'Method', u'Set']

if report_constant:
    plot_cols = ['constant'] + plot_cols

for idx, results in enumerate(method_results):
    if type_of_table == 'metrics by depth':
        results.rename(new_cols_dict, axis=1, inplace=True)
    results = results[plot_cols]
    # for col in plot_cols:
        # if 'p-rule' in col or 'n-rule' in col or 'Accuracy' in col:
        # if 'p-rule' in col or 'n-rule' in col:
        #     results.loc[:, col] = results[col].div(100)
    method_results[idx] = results

# %%
# join all dataframes
report_mat_all_models_df = pd.concat(method_results, axis=0).reset_index()
report_mat_all_models_df.rename({'index': 'sample'}, axis=1, inplace=True)

# %%
if report_constant:
    val_to_nconstant = dict()
    values_hue = list(set(report_mat_all_models_df[hue]))
    for val in values_hue:
        df_hue = report_mat_all_models_df.loc[report_mat_all_models_df[hue] == val]
        val_to_nconstant[val] = np.sum(df_hue['constant'])
    report_mat_all_models_df.drop('constant', inplace=True, axis=1)


# %%
# long format table
dfm = report_mat_all_models_df.melt(id_vars=['sample', hue])
# idx_drop = np.where(dfm['value'] < 0)[0]
# dfm = dfm.drop(idx_drop)
# dfm = dfm.reset_index(drop=True)
# results = results.drop([20, 23, 25])
# results = results.reset_index(drop=True)
# results.iloc[np.where(results < 0)] = 0

# %%
with sns.plotting_context(rc={"legend.fontsize": 18, "legend.title_fontsize": 22}):

    if violin:
        g = sns.catplot(kind='violin', data=dfm, y='variable', x='value', hue=hue, height=12, hue_order=hue_order,
                        palette=palette, legend_out=True).set(xlabel='value')
        g._legend.set_bbox_to_anchor((1, legend_pos))
        plt.grid()
        # plt.xlim([dfm_min, 100])
        locs, labels = plt.yticks()
        plt.setp(labels, rotation=65, fontsize=20)
        locs, labels = plt.xticks()

        plt.setp(labels, fontsize=24)
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')

        # legends
        if hue == 'cov':
            new_title = 'Constraint | # Constant'
            g._legend.set_title(new_title)

            # replace labels
            covs = [e.get_text() for e in g._legend.texts][1:]
            covs = [str(round(float(cov), 4)) for cov in covs]
            new_labels = ['NA (LRT) | # c. = %d' % val_to_nconstant[-1]] + \
                         ['c = %s | # c. = %d' % (cov, val_to_nconstant[float(cov)]) for cov in covs]
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)

        elif hue == 'lambda':
            new_title = 'lambda / n. c.'
            g._legend.set_title(new_title)
            lambdas = [e.get_text() for e in g._legend.texts][1:]
            new_labels = ['NA (Gini)'] + \
                         [r'ļambda = %s ' % lmbda for lmbda in lambdas]
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)
        # elif hue == 'p_epsilon':
        #     new_title = 'p_epsilon / n.c.'
        #     g._legend.set_title(new_title)
        #     g._legend.set_bbox_to_anchor((0.95, .85))
        #     p_epsilons = [e.get_text() for e in g._legend.texts][2:]
        #     # p_epsilons = [e.get_text() for e in g._legend.texts][3:]
        #     # new_labels = ['CF-CARTLR: NA , n.c. %d' % val_to_nconstant[-3]] +\
        #     #              ['CF-TLR: NA , n.c. %d' % val_to_nconstant[-2]] +\
        #     #              ['CF-FCARTLR: NA , n.c. %d' % val_to_nconstant[-1]] +\
        #     #              ['p_e = %s, n.c. %d' % (e, val_to_nconstant[float(e)]) for e in p_epsilons]
        #     new_labels = ['TLR: NA , n.c. %d' % val_to_nconstant[-2]] + \
        #                  ['CF-TLR: NA , n.c. %d' % val_to_nconstant[-1]] + \
        #                  ['p_e = %s, n.c. %d' % (e, val_to_nconstant[float(e)]) for e in p_epsilons]
        #     for t, l in zip(g._legend.texts, new_labels):
        #         t.set_text(l)

        # plt.title('Metrics battery ' + ' '.join([experiment, data_name]))
        plot_name = 'violin plot of: ' + ' '.join([experiment, data_name])
        plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
        plt.show()

    # make boxplot
    if boxplot:
        g = sns.catplot(kind='box', data=dfm, y='variable', x='value', hue=hue, height=12, showfliers=True,
                        hue_order=hue_order, palette=palette,  legend_out=True).set(xlabel='value')

        plt.grid()
        # plt.xlim([dfm_min, 100])
        locs, labels = plt.yticks()
        plt.setp(labels, rotation=65, fontsize=15)
        locs, labels = plt.xticks()
        plt.setp(labels, fontsize=24)
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')

        for ax in g.axes.flat:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

        g._legend.set_bbox_to_anchor((1, legend_pos))


        if hue == 'cov':
            new_title = 'Constraint | # Constant'
            g._legend.set_title(new_title)
            # replace labels
            covs = [e.get_text() for e in g._legend.texts][1:]
            covs = [str(round(float(cov), 4)) for cov in covs]
            if method == 'LRT':
                new_labels = ['NA (LRT) | # c. = %d' % val_to_nconstant[-1]] + \
                             ['c = %s | # c. = %d' % (cov, val_to_nconstant[float(cov)]) for cov in covs]
            else:
                assert method == 'LR'
                new_labels = ['NA (LR)'] + \
                             ['c = %s' % cov for cov in covs]
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)
        elif hue == 'lambda':
            new_title = 'lambda | # Constant'
            g._legend.set_title(new_title)
            # replace labels
            lambdas = [e.get_text() for e in g._legend.texts][1:]
            new_labels = ['NA (MCCP)| # c. = %d' % val_to_nconstant[0.0]] + \
                         ['ļambda = %s | # c. = %d' % (lmbda, val_to_nconstant[float(lmbda)]) for lmbda in lambdas]
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)
        elif hue == 'p-epsilon':
            new_title = 'p-epsilon / n.c.'
            g._legend.set_title(new_title)
            p_epsilons = [e.get_text() for e in g._legend.texts][1:]
            new_labels = ['NA (Gini) | # c. = %d' % val_to_nconstant[-1]] + \
                         ['epsilon = %s | # c. = %d' % (e, val_to_nconstant[float(e)]) for e in p_epsilons]
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)

        plot_name = 'boxplot plot of: ' + ' '.join([experiment, data_name])
        plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
        plt.show()

    # make swarmplot plot with all data point:
    if swarmplot:
        g = sns.swarmplot(x='value', y='variable', data=dfm, hue=hue, hue_order=hue_order, palette=palette)
        locs, labels = plt.yticks()

        # if hue == 'p-epsilon':
        #     p_epsilons = [e.get_text() for e in g.legend.text][3:]
        #     new_labels = ['NA (CF-CARTLR)'] +\
        #                  ['NA (CF-FCARTLR)'] + \
        #                  ['NA (CF-TLR)'] + \
        #                  ['p_e = %s' % e for e in p_epsilons]
        #     g.legend(labels=new_labels)

        plt.setp(labels, rotation=45, fontsize=10)
        plt.grid()
        plt.title('Metrics battery ' + ' '.join([experiment, data_name]))
        plot_name = 'swarmplot plot of: ' + ' '.join([experiment, data_name])
        plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
    plt.show()

    # distribution plots of each metric
    if density:
        rows = 1
        columns = 4
        fig, axes = plt.subplots(rows, columns)
        if hue_order is not None:
            labels = list(hue_order)
            hue_order_dens = list(hue_order)
        else:
            lambdas = report_mat_all_models_df[hue].dropna()
            lambdas = list(set(list(lambdas)))
            lambdas.sort()
            labels = lambdas
            hue_order_dens = lambdas
        # hue_order_dens
        for n_sub_fig, x in enumerate(['p-rule', 'n-rule', 'SP', 'Accuracy']):
            d= sns.kdeplot(data=report_mat_all_models_df, x=x, hue=hue, hue_order=hue_order_dens,
                        ax=axes[n_sub_fig], palette=None)
            axes[n_sub_fig].label_outer()
            # plt.setp(d.get_yticklabels(), fontsize=5,
            #          horizontalalignment="left")

            # if x == 'p-rule':
            #     axes[n_sub_fig].set_xlim([0, 100])
            #     # axes[n_sub_fig].set_ylim([0, .025])
            # else:
            #     d.set_ylabel('')
                # d.get_legend().remove()
            if x != 'p-rule':
                d.set_ylabel('')
            d = sns.rugplot(data=report_mat_all_models_df, x=x, hue=hue, hue_order=hue_order_dens,
                        palette=None, ax=axes[n_sub_fig])
            axes[n_sub_fig].label_outer()
            plt.setp(d.get_yticklabels(), fontsize=5,
                     horizontalalignment="left")

            d.get_legend().remove()

        fig.tight_layout()
        # labels.reverse()
        fig.legend(labels=labels, loc="upper left", ncol=1, title='lambda', prop={'size': 10})

        plot_name = 'density plot of: ' + ' '.join([experiment, data_name])
        plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
    plt.show()

    # hitograms of each metric
    if histogram and type_of_table == 'metrics':
        rows = 1
        columns = 4
        fig, axes = plt.subplots(rows, columns)
        labels = list(hue_order)
        labels.reverse()
        hue_order_dens = list(hue_order)
        for n_sub_fig, x in enumerate(['p-rule', 'n-rule', 'SP', 'Accuracy']):
        # for n_sub_fig, x in enumerate(['n terminals']):
            d = sns.histplot(data=report_mat_all_models_df, x=x, hue="Method", hue_order=hue_order_dens,
                         ax=axes[n_sub_fig], palette=palette)
            axes[n_sub_fig].label_outer()
            plt.setp(d.get_yticklabels(), fontsize=5,
                     horizontalalignment="left")
            d.get_legend().remove()
            if n_sub_fig == 0:
                axes[n_sub_fig].set_xlim([0, 100])
            else:
                d.set_ylabel('')

        fig.tight_layout()
        fig.legend(labels=labels, loc="upper left", ncol=1, title='Method', prop={'size': 10})

        plot_name = 'histogram of: ' + ' '.join([experiment, data_name])
        plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
        plt.show()

    if histogram and type_of_table == 'terminals':
        labels = list(hue_order)
        labels.reverse()
        hue_order_dens = list(hue_order)
        x = 'n terminals'
        d = sns.histplot(data=report_mat_all_models_df, x=x, hue="Method", hue_order=hue_order_dens,
                         palette=palette)
        d.label_outer()
        # plt.setp(d.get_yticklabels(), fontsize=5,
        #          horizontalalignment="left")
        # d.get_legend().remove()

        # fig.tight_layout()
        # fig.legend(labels=labels, loc="upper left", ncol=1, title='Method', prop={'size': 10})

        plot_name = 'histogram of: ' + ' '.join([experiment, data_name])
        plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
        plt.show()


#%%
