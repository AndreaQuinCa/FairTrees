import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns

# %%
# parameters
experiment = 'LRT_C-LRT'
data_name = 'compas'
hue = 'Method'
hue_order = ['LRT', 'C-LRT']
type_of_table = 'variable type'

# get filenames
folder_specific_experiment = "to_compare_" + experiment
folder_experiments = "/".join(['tables', folder_specific_experiment, data_name, type_of_table])
folder_experiments_img = "/".join([folder_experiments, 'img'])
folder_experiments_csv = "/".join([folder_experiments, '*.csv'])
filenames = glob.glob(folder_experiments_csv)


# load csv's as dataframes
n_methods = len(filenames)
method_results = [0] * n_methods
for i, filename in enumerate(filenames):
    method_results[i] = pd.read_csv(filename, sep=',')

plot_cols_cases = method_results[0].columns
plot_cols_cases = [e for e in plot_cols_cases if 'Case']
plot_cols_counts = method_results[0].columns
plot_cols_counts = [e for e in plot_cols_counts if 'Case' not in e]
plot_cols_counts_discrete = method_results[0].columns
plot_cols_counts_discrete = [e for e in plot_cols_counts_discrete if 'Case' not in e and 'iscrete' in e]
plot_cols_counts_continuos = method_results[0].columns
plot_cols_counts_continuos = [e for e in plot_cols_counts_continuos if 'Case' not in e and 'ontinuous' in e]
max_depth = len(plot_cols_counts_discrete)-1

plot_cols = plot_cols_counts
# results = method_results[0]

#%%
for idx, results in enumerate(method_results):
    results = results[plot_cols]
    # results = results.drop([20, 23, 25])
    results = results.reset_index(drop=True)
    # results.iloc[np.where(results < 0)] = 0
    method_results[idx] = results


#%%
# convert counts to percentage and add total nodes per depth
for results in method_results:
    new_cols = ['Nodes in depth %d' % d for d in range(max_depth+1)]
    for col_name in new_cols:
        results[col_name] = np.NAN

    for row_index, row in results.iterrows():
        row_list = row.tolist()[:len(plot_cols)-1]
        max_depth = int((len(row_list)-1) / 2)
        total_node_per_depth = [0] * int(max_depth+1)
        idx_d = 0
        for d in range(max_depth):
            total_node_per_depth[d] = row_list[idx_d] + row_list[idx_d+1]
            if total_node_per_depth[d] > 0:
                row_list[idx_d] = row_list[idx_d]/total_node_per_depth[d]
                row_list[idx_d+1] = row_list[idx_d+1]/total_node_per_depth[d]
            idx_d = idx_d + 2
        results.iloc[row_index, 0:len(row_list)] = row_list

        assert (len(new_cols) == len(total_node_per_depth) == max_depth+1)
        for depth, col_name in enumerate(new_cols):
            results.loc[row_index, col_name] = total_node_per_depth[depth]

#%%
plot_cols_sums = ['Depth', 'Mean % discrete', 'Mean % continuous']
depths = list(range(max_depth+1))

results_means = [0]*len(method_results)
for idx, results in enumerate(method_results):
    means = results.mean()

    mean_nodes = np.array([means[e] for e in means.keys() if 'Nodes' in e])

    per_discrete = np.array([means[e] for e in means.keys() if 'iscrete' in e])
    per_discrete = per_discrete*mean_nodes

    per_continuous = np.array([means[e] for e in means.keys() if 'ontinuous' in e])
    per_continuous = per_continuous*mean_nodes

    results_means[idx] = pd.DataFrame({'Depth': depths,
             'Mean % discrete': per_discrete,
             'Mean % continuous': per_continuous})


#%%

# join all dataframes
results_means_all_models_df = pd.concat(results_means, axis=0).reset_index()
results_means_all_models_df.rename({'index': 'sample'}, axis=1, inplace=True)
results_means_all_models_df = results_means_all_models_df.drop(6)
#%%
df = results_means_all_models_df[['Depth', 'Mean % discrete', 'Mean % continuous']]
df.set_index('Depth').plot(kind='bar', stacked=True, color=['m', 'y'])
plt.xlabel('Depth')
plt.ylabel('Mean number of decision nodes')
plot_name = 'Barplot: ' + ' '.join([experiment, type_of_table, data_name])
plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
plt.show()
#%%

def compare_metrics(report_mat_all_models_df, name_plot_cols):

    var_type = name_plot_cols.keys()[0]
    plot_cols = name_plot_cols.values()[0]
    report_mat_all_models_df = report_mat_all_models_df[plot_cols + ['sample']]
    dfm = report_mat_all_models_df.melt(id_vars=['sample', hue])

    # histogram with all data point: y='variable',
    sns.barplot(data=dfm, x='variable', y='value', hue=hue, hue_order=hue_order).set(
        xlabel='value')
    plt.grid()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45, fontsize=5)
    plt.title(' '.join([experiment, data_name]))
    plot_name = 'Barplot of: ' + ' '.join([experiment, type_of_table, var_type, data_name])
    plt.savefig(folder_experiments_img + '/' + plot_name + '.png')
    plt.show()




#%%
