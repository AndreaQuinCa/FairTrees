import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data_utils import add_path, encode_data, divide_df, add_intercept, get_one_hot_encoding, get_sample, split_data
from evaluations_fairness import calculate_performance_fair_mine, get_confusion_matrices, get_balanced_accuracy,\
    calculate_performance_sp

#%%

def get_parameters(filename_parameters, data_name, my_path):
    # load parameters file
    parameter_folder = '/'.join(['tables', 'parameters', data_name])
    parameter_folder = add_path(parameter_folder, my_path)
    parameters_experiments = pd.read_csv(parameter_folder + '/' + filename_parameters, sep=',')
    return parameters_experiments



def format_parameters(parameter, s_attr):

    method_name = parameter['method_name']
    print('\n' + method_name)

    # cov
    if 'C-' not in method_name:
        apply_fairness_constraints = 0
        assert (parameter['cov'] == 'None')
        cov = None
    else:
        apply_fairness_constraints = 1
        assert (parameter['cov'] != 'None')
        cov = float(parameter['cov'])
        print('cov:', cov)
    sensitive_attrs_to_cov_thresh = {s_attr: cov}

    # prune parameters
    fmccp = 'FMCCP' in method_name
    if fmccp:
        assert (parameter['lambda'] != 'None')
        lmbda = float(parameter['lambda'])
        assert 0. <= lmbda <= 1.
        print('lambda:', lmbda)
    else:
        lmbda = None

    relab = 'RELAB' in method_name
    if relab:
        assert (parameter['p_epsilon'] != 'None')
        p_epsilon = float(parameter['p_epsilon'])
        assert 0. <= p_epsilon <= 1.
        print('p_epsilon:', p_epsilon)
    else:
        p_epsilon = None

    mccp = 'MCCP' in method_name and 'FMCCP' not in method_name

    return method_name, apply_fairness_constraints, sensitive_attrs_to_cov_thresh, cov, mccp, fmccp, relab, lmbda, \
           p_epsilon


def prepare_data(data_name, my_path, add_intercept_bool):

    # load data
    if 'adult' in data_name:
        loading_fun = load_adult
    elif 'kdd' in data_name:
        loading_fun = load_kdd
    elif 'german' in data_name:
        loading_fun = load_german
    elif 'dutch' in data_name:
        loading_fun = load_dutch
    elif 'bank' in data_name:
        loading_fun = load_bank
    elif 'credit' in data_name:
        loading_fun = load_credit
    elif 'compas' in data_name:
        loading_fun = load_compas_recid
    elif 'crime' in data_name:
        loading_fun = load_crime
    elif 'diabetes' in data_name:
        loading_fun = load_diabetes
    elif 'ricci' in data_name:
        loading_fun = load_ricci
    elif 'student_por' in data_name:
        loading_fun = load_student_por
    elif 'student_mat' in data_name:
        loading_fun = load_student_mat
    elif 'oulad' in data_name:
        loading_fun = load_oulad
    elif 'law' in data_name:
        loading_fun = load_law
    else:
        raise Exception('"%s" is not a valid data name')

    continuous = 'continuous' in data_name
    if 'dutch' in data_name or 'crime' in data_name:
        if continuous:
            raise Exception("In %s dataset continuous and not continuous are identycal" % data_name)
        df, protected_attribute, class_label, numeric_attrs_idx_set = loading_fun(my_path)
    else:
        df, protected_attribute, class_label, numeric_attrs_idx_set = loading_fun(continuous, my_path)

    X, y, sensitive = divide_df(df, protected_attribute, class_label)

    if add_intercept_bool:
        X = add_intercept(X)
        if numeric_attrs_idx_set is not None:
            numeric_attrs_idx_set = set([e + 1 for e in numeric_attrs_idx_set])

    protected_attributes = [protected_attribute]
    n_sample = X.shape[0]
    if n_sample > 10000:
        n_sample = 10000

    return X, y, sensitive, numeric_attrs_idx_set, protected_attribute, protected_attributes, n_sample


#%%

def load_adult(continuous=False, my_path=None, one_hot=True):

    # data parameters
    filename = 'adult-clean.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    protected_attribute = 'gender'
    protected_group_name = 'Female'
    class_label = 'Class-label'
    advantageous_label_value = 1

    # load data frame
    df = pd.read_csv(filename)

    # Tranform categorical attributes to reduce number of categories
    df['workclass'] = ['private' if v == 'Private' else 'non-private' for v in df['workclass']]
    high_degree = ['Prof-school', 'Assoc-voc', 'Assoc-acdm', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
    df['education'] = ['high' if v in high_degree else 'low' for v in df['education']]
    married = ['Married-AF-spouse', 'Married-civ-spouse']
    df['marital-status'] = ['married' if v in married else 'other' for v in df['marital-status']]
    relationship = ['Husband', 'Wife']
    df['relationship'] = ['married' if v in relationship else 'other' for v in df['relationship']]
    df['native-country'] = ['US' if v == 'United-States' else 'non-US' for v in df['native-country']]
    office = ['Exec-managerial', 'Adm-clerical', 'Sales', 'Tech-support']
    heavy_work = ['Craft-repair', 'Farming-fishing', 'Transport-moving',
                  'Machine-op-inspct']
    df['occupation'] = ['office' if v in office else 'heavy_work' if v in heavy_work else 'other'
                        for v in df['occupation']]

    # Filter columns
    if not continuous:
        new_columns = ['age', 'workclass', 'education', 'educational-num', 'marital-status', 'occupation',
                       'relationship', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                       'Class-label']
    else:
        new_columns = ['age', 'educational-num', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'Class-label']
    df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_kdd(continuous=False, my_path=None, one_hot=True):

    # data parameters
    filename = 'kdd-census-income-clean.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    protected_attribute = 'sex'
    protected_group_name = 'Female'
    class_label = 'income'
    advantageous_label_value = 1

    # load data frame
    df = pd.read_csv(filename)

    # format rows
    df['income'] = [1 if v == ">50K" else 0 for v in df['income']]

    # Format columns
    col_names = list(df.dtypes.index)
    numeric_cols = ['age', 'wage-per-hour', 'capital-gain', 'capital-loss', 'dividends-from-stocks',
                    'num-persons-worked-for-employer', 'weeks-worked']
    cols_to_format = dict()
    for col_name in col_names:
        if col_name in numeric_cols:
            cols_to_format[col_name] = 'int64'
        else:
            cols_to_format[col_name] = 'object'
    df = df.astype(cols_to_format)
    # Filter columns
    if continuous:
        new_columns = numeric_cols + [protected_attribute, class_label]
        df = df[new_columns]
    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_german(continuous=False, my_path=None, one_hot=True):

    # data parameters
    filename = 'german_data_credit.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    protected_attribute = 'sex'
    protected_group_name = 'female'
    class_label = 'class-label'
    advantageous_label_value = 1

    # load data frame
    df = pd.read_csv(filename)

    # Filter columns
    if continuous:
        new_columns = ['duration', 'sex', 'credit-amount', 'installment-rate', 'residence-since', 'age',
                       'existing-credits', 'numner-people-provide-maintenance-for', 'class-label']
        df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_dutch(my_path=None, one_hot=True):

    # data parameters
    filename = 'dutch.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    protected_attribute = 'sex'
    protected_group_name = 'female'
    class_label = 'occupation'
    advantageous_label_value = 1

    # load data frame
    df = pd.read_csv(filename)

    # format type (all attributes are categorical)
    df = df.astype('object')

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous=False)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_bank(continuous=False, my_path=None, one_hot=True):

    # data parameters
    filename = 'bank-additional-full.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    protected_attribute = 'age'  # or marital-status
    protected_group_name = 'non-middle-age'
    class_label = 'y'
    advantageous_label_value = 'yes'

    # load data frame
    df = pd.read_csv(filename, delimiter=';')

    # Format protected attr
    df['age'] = ['middle-age' if 25 <= v <= 60 else 'non-middle-age' for v in df['age']]

    # Filter columns
    df.drop('duration', axis=1, inplace=True)
    if continuous:
        new_columns = ["campaign", "age", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                       "euribor3m", "nr.employed", "y"]
        df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous=continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_credit(continuous=False, my_path=None, one_hot=True):
    # data parameters
    filename = 'credit-card-clients.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    protected_attribute = 'SEX'  # marriage or education
    protected_group_name = 2
    class_label = 'default payment'
    advantageous_label_value = 0

    # load data frame
    df = pd.read_csv(filename)

    # Format columns
    cols_to_format_list = ['EDUCATION', 'MARRIAGE'] + ['PAY_0'] + ['PAY_' + str(n + 2) for n in range(5)]
    cols_to_format = dict()
    for k in cols_to_format_list:
        cols_to_format[k] = 'object'
    df = df.astype(cols_to_format)

    # Filter columns
    if continuous:
        new_columns = ['LIMIT_BAL', 'SEX', 'AGE'] + ['BILL_AMT'+str(n + 1) for n in range(6)] +\
                      ['PAY_AMT' + str(n + 1) for n in range(6)] + ['default payment']
        df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


#%%

def load_compas_recid(continuous=False, my_path=None, one_hot=True):
    filename = 'compas-scores-two-years_clean.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    df = pd.read_csv(filename)
    protected_attribute = 'race'
    protected_group_name = 'African-American'
    class_label = 'two_year_recid'
    advantageous_label_value = 0

    # Remove all row without black and white
    df = df[(df['race'] == 'African-American') | (df['race'] == 'Caucasian')]

    # Filter columns
    if not continuous:
        new_columns = ["age", "race", "sex", "age_cat", "priors_count", "decile_score", "juv_fel_count",
                       "juv_misd_count", "juv_other_count", "c_charge_degree", "score_text", "v_score_text",
                       "two_year_recid"]
    else:
        new_columns = ["age", "race", "priors_count", "juv_fel_count", "juv_misd_count",
                       "juv_other_count", "two_year_recid"]

    df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_crime(my_path=None, one_hot=True):

    # data parameters
    filename = 'communities_crime.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    protected_attribute = 'Black'
    protected_group_name = 1
    class_label = 'class'
    advantageous_label_value = 0

    # load data frame
    df = pd.read_csv(filename)

    # remove ids
    df.drop(['fold', 'communityname'], inplace=True, axis=1)

    # format cols
    cols_to_format = {'state': 'object'}
    df = df.astype(cols_to_format)

    # encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous=False)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_diabetes(continuous=False, my_path=None, one_hot=True):
    filename = 'diabetes-clean.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    df = pd.read_csv(filename)
    protected_attribute = 'gender'
    protected_group_name = 'Female'
    class_label = 'readmitted'
    advantageous_label_value = 1

    # Format rows
    df['readmitted'] = [1 if v == '<30' else 0 for v in df['readmitted']]

    # Filter columns
    if continuous:
        df.drop('Position', inplace=True, axis=1)

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_ricci(continuous=False, my_path=None, one_hot=True):
    filename = 'ricci_race.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    df = pd.read_csv(filename)
    protected_attribute = 'Race'
    protected_group_name = 'Non-White'
    class_label = 'Promoted'
    advantageous_label_value = 1

    # Filter columns
    if continuous:
        df.drop('Position', inplace=True, axis=1)

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_student_mat(continuous=False, my_path=None, one_hot=True):
    filename = 'student_mat_new.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    df = pd.read_csv(filename)
    protected_attribute = 'sex'
    protected_group_name = 'F'
    class_label = 'class'
    advantageous_label_value = 'High'

    # Filter columns
    if continuous:
        new_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                       'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
                       'G1', 'G2', protected_attribute, class_label]
        df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_student_por(continuous=False, my_path=None, one_hot=True):
    filename = 'student_por_new.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    df = pd.read_csv(filename)
    protected_attribute = 'sex'
    protected_group_name = 'F'
    class_label = 'class'
    advantageous_label_value = 'High'

    # Filter columns
    if continuous:
        new_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                       'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
                       'G1', 'G2', protected_attribute, class_label]
        df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_oulad(continuous=False, my_path=None, one_hot=True):
    filename = 'oulad_clean.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    df = pd.read_csv(filename)
    protected_attribute = 'gender'
    protected_group_name = 'F'
    class_label = 'final_result'
    advantageous_label_value = 'Pass'

    # Filter columns
    df.drop('id_student', inplace=True, axis=1)
    if continuous:
        new_columns = ['gender', 'num_of_prev_attempts', 'studied_credits', 'final_result']
        df = df[new_columns]

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


def load_law(continuous=False, my_path=None, one_hot=True):
    filename = 'law_school_clean.csv'
    filename = add_path('datasets/' + filename, my_path=my_path)
    df = pd.read_csv(filename)
    protected_attribute = 'race'  # or male
    protected_group_name = 'Non-White'
    class_label = 'pass_bar'
    advantageous_label_value = 1

    # format columns
    cols_to_format_list = ['fulltime', 'fam_inc', 'male', 'tier']
    cols_to_format = dict()
    for k in cols_to_format_list:
        cols_to_format[k] = 'object'
    df = df.astype(cols_to_format)

    # Filter columns
    if continuous:
        df.drop(['fulltime', 'fam_inc', 'tier'], inplace=True, axis=1)

    # Encode data
    numeric_attrs_idx_set = None
    if one_hot:
        df, numeric_attrs_idx_set = encode_data(df, protected_attribute, protected_group_name, class_label,
                                            advantageous_label_value, continuous)

    return df, protected_attribute, class_label, numeric_attrs_idx_set


#%%
def get_labels(data):
    if data == 'adult':
        protected_group_name = 'Female'
        non_protected_group_name = 'Male'
        advantageous_label_value = 1
        disadvantageous_label_value = 0
        fun = load_adult
    if data == 'kdd':
        protected_group_name = 'Female'
        non_protected_group_name = 'Male'
        advantageous_label_value = 1
        disadvantageous_label_value = 0
        fun = load_kdd
    elif data == 'german':
        protected_group_name = 'female'
        non_protected_group_name = 'male'
        advantageous_label_value = 1
        disadvantageous_label_value = 0
        fun = load_german
    elif data == 'dutch':
        protected_group_name = 'female'
        non_protected_group_name = 'male'
        advantageous_label_value = 1
        disadvantageous_label_value = 0
        fun = load_dutch
    elif data == 'bank':
        protected_group_name = 'non-middle-age'
        non_protected_group_name = 'middle-age'
        advantageous_label_value = 'yes'
        disadvantageous_label_value = 'no'
        fun = load_bank
    elif data == 'credit':
        protected_group_name = 2
        non_protected_group_name = 1
        advantageous_label_value = 0
        disadvantageous_label_value = 1
        fun = load_credit
    elif data == 'compas':
        protected_group_name = 'African-American'
        non_protected_group_name = 'Caucasian'
        advantageous_label_value = 0
        disadvantageous_label_value = 1
        fun = load_compas_recid
    elif data == 'crime':
        protected_group_name = 1
        non_protected_group_name = 0
        advantageous_label_value = 0
        disadvantageous_label_value = 1
        fun = load_crime
    elif data == 'diabetes':
        protected_group_name = 'Female'
        non_protected_group_name = 'Male'
        advantageous_label_value = 1
        disadvantageous_label_value = 0
        fun = load_diabetes
    elif data == 'ricci':
        protected_group_name = 'Non-White'
        non_protected_group_name = 'White'
        advantageous_label_value = 1
        disadvantageous_label_value = -1
        fun = load_ricci
    elif data == 'student_por':
        protected_group_name = 'F'
        non_protected_group_name = 'M'
        advantageous_label_value = 'High'
        disadvantageous_label_value = 'Low'
        fun = load_student_por
    elif data == 'student_mat':
        protected_group_name = 'F'
        non_protected_group_name = 'M'
        advantageous_label_value = 'High'
        disadvantageous_label_value = 'Low'
        fun = load_student_mat
    elif data == 'oulad':
        protected_group_name = 'F'
        non_protected_group_name = 'M'
        advantageous_label_value = 'Pass'
        disadvantageous_label_value = 'Fail'
        fun = load_oulad
    elif data == 'law':
        protected_group_name = 'Non-White'
        non_protected_group_name = 'White'
        advantageous_label_value = 1
        disadvantageous_label_value = 0
        fun = load_law
    return fun, protected_group_name, non_protected_group_name, advantageous_label_value, disadvantageous_label_value


#%%
if __name__ == '__main__':
    # Parámetros dependientes de conjunto
    data = 'law'
    do_exp = False
    big_experiment = True
    #%%
    load_fun, protected_group_name, non_protected_group_name, advantageous_label_value, disadvantageous_label_value = get_labels(
        data)

    # Carga de datos
    df, protected_attribute, class_label, numeric_attrs_idx_set = load_fun(one_hot=False)

    #%%
    # Cantidad de instancias
    total = df.shape[0]
    print("\n" + "-" * 100)
    print('total', total)

    # Missing values
    print("\n" + "-" * 100)
    print('Missing values')
    for i, j in zip(df.columns, (df.values.astype(str) == 'N/A').sum(axis=0)):
        if j > 0:
            print(str(i) + ': ' + str(j) + ' records')
    print(df.isnull().sum())

    #%%
    # Tipo de atributos
    col_types = df.dtypes
    print("\n" + "-" * 100)
    print('col_types')
    print()
    print('  counts:')
    print(col_types.value_counts())
    print('  list:')
    print(col_types)

    #%%
    # proporción de clases Y=y
    df_pos = df[df[class_label] == advantageous_label_value]
    n_pos = df_pos.shape[0]

    df_neg = df[df[class_label] == disadvantageous_label_value]
    n_neg = df_neg.shape[0]
    print("\n" + "-" * 100)
    min_n = min(float(n_neg), float(n_pos))
    max_n = max(float(n_neg), float(n_pos))
    print('IR', '1:%0.2f' % (float(max_n) / float(min_n)))

    all_y = df[class_label].value_counts() / total
    print('\n')
    print(all_y)

    # proporción de grupos A=a
    df_prot = df[df[protected_attribute] == protected_group_name]
    n_prot = df_prot.shape[0]
    prop_prot = float(n_prot) * 100 / total

    df_non_prot = df[df[protected_attribute] == non_protected_group_name]
    n_non_prot = df_non_prot.shape[0]
    prop_non_prot = float(n_non_prot) * 100 / total
    print("\n" + "-" * 100)
    print('protected rate', '%0.1f:%0.1f' % (prop_non_prot, prop_prot))

    all_a = df[protected_attribute].value_counts() / total
    print('\n')
    print(all_a)

    #%%
    # tabla de contigencia
    contingency_tab = pd.crosstab(df[class_label], df[protected_attribute])
    contingency_tab = contingency_tab[[protected_group_name,non_protected_group_name]]
    print("\n" + "-" * 100)
    print('Contigency table')
    print(contingency_tab.round(3))
    print("\n")
    print('Contigency normalized by A')
    tab_names = contingency_tab.columns
    print(contingency_tab.div([n_prot, n_non_prot], 1).round(3) * 100)


    # %%
    # codificación estándar de Y and A
    df_one_hot, cont = encode_data(df, protected_attribute, protected_group_name, class_label, advantageous_label_value, False)
    dtypess = df_one_hot.dtypes
    a_encode = [(org, enc) for org, enc in zip(df[protected_attribute], df_one_hot[protected_attribute])][0:30]
    y_encode = [(org, enc) for org, enc in zip(df[class_label], df_one_hot[class_label])][0:30]
    print("\n" + "-" * 100)
    print('A encoding')
    print(a_encode)

    print("\n" + "-" * 100)
    print('Y encoding')
    print(y_encode)

    # %%
    # dividir atributos
    X, y, sensitive = divide_df(df_one_hot, protected_attribute, class_label)
    sensitive = sensitive[protected_attribute]

    # %%
    if do_exp:
        n_reps = 5
        accs = np.zeros(n_reps) + 2
        stat_pars = np.zeros(n_reps) + 2
        for rep in range(n_reps):
            X_train, X_test, sensitive_train, sensitive_test, y_train, y_test = train_test_split(X, sensitive, y, test_size=0.3,
                                                                                                 random_state=42 + rep * 2)
            LR = LogisticRegression(solver='liblinear', random_state=42)
            LR.fit(X_train, y_train)
            y_predicts = LR.predict(X_test)
            y_pred_probs = LR.predict_proba(X_test)

            # Paridad Estadística
            protected_group_name = 0
            stat_par, accuracy = calculate_performance_sp(y_test, y_predicts, sensitive_test, protected_group_name)
            accs[rep] = accuracy
            stat_pars[rep] = stat_par

        print("\n" + "-" * 100)
        print('LR stats')
        print('acc', 'mean:', round(accs.mean(), 4), ', var:', round(accs.var(), 4))
        print('sp', 'mean:', round(stat_pars.mean(), 4), ', var:', round(stat_pars.var(), 4))

    #%%
    from random import seed
    if big_experiment:
        # Parámetros dependientes de conjunto
        continuous = False
        datasets = ['adult', 'compas', 'ricci', 'law']
        # datasets = ['ricci']
        # datasets = ['adult', 'kdd', 'german', 'bank', 'credit',  'compas', 'ricci', 'student_por', 'student_mat',
        #             'oulad', 'law']
        # if continuous:
        #     datasets = [data + ' continuous' for data in datasets]
        # else:
        #     datasets = datasets + ['dutch', 'crime']

        n_reps = 30
        val_size = 0.2
        test_size = 0.2
        SEED = 110979
        seed(SEED)
        np.random.seed(SEED)
        results_df = []

        for data in datasets:

            print('\n'+'-'*30)
            print(data)

            #Load and pre-proces data
            x, y, sensitive, continous_idx_set, s_attr, sensitive_attrs, n_sample = prepare_data(data, None, False)

            accs = np.zeros(n_reps)
            stat_pars = np.zeros(n_reps)
            b_accs = np.zeros(n_reps)
            b_accs_prot = np.zeros(n_reps)
            b_accs_non_prot = np.zeros(n_reps)
            p_rules = np.zeros(n_reps)
            n_rules = np.zeros(n_reps)

            for rep in range(n_reps):
                print('rep', rep)
                rep_seed = SEED * (rep + 1)
                x_sample, y_sample, sensitive_sample = get_sample(x, y, sensitive, rep_seed, n_sample, do_shuffle=True)

                x_train, x_val, x_test, sensitive_train, sensitive_val, sensitive_test, y_train, y_val, y_test = \
                    split_data(x_sample, sensitive_sample, y_sample, s_attr, val_size=val_size, test_size=test_size)

                LR = LogisticRegression(solver='liblinear', random_state=42)
                LR.fit(x_train, y_train)

                y_predicts = LR.predict(x_val)
                sensitive_val_arr = sensitive_val[s_attr]
                stat_par, p_rule, n_rule = calculate_performance_fair_mine(y_predicts, sensitive_val_arr)
                accs[rep] = accuracy_score(y_val, y_predicts)
                stat_pars[rep] = stat_par
                cm, cm_prot, cm_non_prot = get_confusion_matrices(y_val, y_predicts, sensitive_val_arr)
                b_accs[rep] = get_balanced_accuracy(cm)
                b_accs_prot[rep] = get_balanced_accuracy(cm_prot)
                b_accs_non_prot[rep] = get_balanced_accuracy(cm_non_prot)
                p_rules[rep] = p_rule
                n_rules[rep] = n_rule

            # initialize data of lists.
            data_results = {'Accuracy': accs,
                            'SP': stat_pars,
                            'p-rule': p_rules,
                            'n-rule': n_rules,
                            'B. accuracy': b_accs,
                            'B. accuracy prot': b_accs_prot,
                            'B. accuracy non-prot': b_accs_non_prot}

            data_results = pd.DataFrame(data_results).round(4)
            data_results.to_csv('tables/experiments LR/' + 'LR %d reps ' % n_reps + data + ' val' + '.csv', index=False)

            data_row = {'Dataset': data,
                        'Mean accuracy': accs.mean(),
                        'Accuracy variance': accs.var(),
                        'Mean SP': stat_pars.mean(),
                        'Absolute SP mean': abs(stat_pars.mean()),
                        'SP variance': stat_pars.var(),
                        'Mean p-rule': p_rules.mean(),
                        'p-rule variance': p_rules.var(),
                        'Mean n-rule': n_rules.mean(),
                        'n-rule variance': n_rules.var(),
                        'Mean b. accuracy': b_accs.mean(),
                        'B. accuracy variance': b_accs.var(),
                        'Mean b. accuracy prot': b_accs_prot.mean(),
                        'B. accuracy prot variance': b_accs_prot.var(),
                        'Mean b. accuracy non-prot': b_accs_non_prot.mean(),
                        'B. accuracy non-prot variance': b_accs_prot.var()}

            results_df.append(data_row)
#%%
        results_df = pd.DataFrame(results_df).round(4)
        results_df.sort_values(by=['Mean SP', 'Mean accuracy'], inplace=True, ascending=False)
        results_df.reset_index(drop=True, inplace=True)
        print(results_df)
        if continuous:
            results_df.to_csv('tables/experiments LR/all_datasets_continuous.csv', index=False)
        else:
            results_df.to_csv('tables/experiments LR/all_datasets.csv', index=False)

