# encoding=utf-8
import numpy as np
from random import seed, shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd


# %%

class DatasetExt:
    """
    Class for managing training and test data information of decisions trees.
    """

    def __init__(self, x_train=None, y_train=None, sensitive_train=None,
                 x_test=None, y_test=None, sensitive_test=None,
                 x_val=None, y_val=None, sensitive_val=None,
                 intercept=True, aprioris=None, costs=None, classes_names=None,
                 classes_sizes=None, sensitive_attrs=None, sensitive_aprioris=None, sensitive_names=None,
                 sensitive_sizes=None):

        self.x_train, self.y_train,  self.sensitive_train = x_train, y_train, sensitive_train
        self.x_test, self.y_test,  self.sensitive_test = x_test, y_test, sensitive_test
        self.x_val, self.y_val,  self.sensitive_val = x_val, y_val, sensitive_val
        self.intercept = intercept
        self.aprioris = aprioris
        self.costs = costs
        self.classes_names = classes_names
        self.classes_sizes = classes_sizes
        self.sensitive_attrs = sensitive_attrs
        self.sensitive_aprioris = sensitive_aprioris
        self.sensitive_names = sensitive_names
        self.sensitive_sizes = sensitive_sizes

        assert self.x_train is not None or self.x_test is not None

        if x_train is None:
            train = False
        else:
            train = True

        if train:
            x = self.x_train
            y = self.y_train
            sensitive = self.sensitive_train
        else:
            x = self.x_test
            y = self.y_test
            sensitive = self.sensitive_test

        self.n_predictors = x.shape[1]
        self.training_size = x.shape[0]
        self.predictors_idx = np.array(range(self.n_predictors), dtype=np.int)
        if self.classes_names is None:
            self.classes_names, self.classes_sizes = np.unique(y, return_counts=True)
        self.classes_names = tuple(self.classes_names)
        self.classes_sizes = tuple(self.classes_sizes)
        self.classes_to_idx = {name: idx for idx, name in enumerate(self.classes_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.classes_names)}
        self.n_classes = len(self.classes_names)
        if self.intercept:  # used for models with intercept
            self.n_predictors = self.n_predictors - 1
            self.predictors_idx = np.array(range(self.n_predictors), dtype=np.int)
            self.predictors_idx = self.predictors_idx + 1
        if self.sensitive_attrs is None:
            self.sensitive_attrs = sorted(list(sensitive.keys()))
        if self.sensitive_names is None:
            sensitive_vals = sensitive[self.sensitive_attrs[0]]
            self.sensitive_names, self.sensitive_sizes = np.unique(sensitive_vals, return_counts=True)
            assert (len(self.sensitive_names) == 2)
        self.sensitive_names = tuple(self.sensitive_names)
        self.sensitive_sizes = tuple(self.sensitive_sizes)
        self.sensitive_groups_to_idx = {name: idx for idx, name in enumerate(self.sensitive_names)}
        self.idx_to_sensitive_groups = {idx: name for idx, name in enumerate(self.sensitive_names)}
        self.n_sensitive = len(self.sensitive_names)
        if self.aprioris is None:
            self.aprioris = np.array(self.classes_sizes, dtype=np.float) / (self.training_size * 1.0)
        if self.sensitive_aprioris is None:
            self.sensitive_aprioris = np.array(self.sensitive_sizes, dtype=np.float) / (self.training_size * 1.0)
        if self.costs is None:  # symmetric cost
            self.costs = np.ones(shape=(self.n_classes, self.n_classes))
            np.fill_diagonal(self.costs, 0)


# %%

# Data transfomation

def add_random_dimensions(X, n_random_dims, m_seed=11, var=3):
    n_X = X.shape[0]
    for n_arr in range(n_random_dims):
        np.random.seed(m_seed * (1 + n_arr))
        random_arr = np.random.uniform(-var, var, (n_X, 1))
        X = np.hstack((X, random_arr))
    return X


def add_intercept(x):
    """
    This function is used from arXiv:1507.05259 [stat.ML] source code. Source link below.

    ----------
    Parameters
    
    Add intercept to the data before linear classification.
    param x: ndarray, shape (n_samples, n_features). Training data.
    return:  ndarray, shape (n_samples, n_features+1). Concatenated intercept 
    is an extra column of ones in the first entry.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git
    
    """
    m, n = x.shape
    intercept = np.ones(m).reshape(m, 1)  # the constant b
    return np.concatenate((intercept, x), axis=1)


def add_path(filename, my_path=None):
    if my_path:
        filename = my_path + '/' + filename
    return filename


# %%

# Data verification

def check_two_sensible_classes(x_sensitive_train, sensitive_attrs):
    two_sensible_classes_all = dict()
    for attr in sensitive_attrs:
        attr_arr = x_sensitive_train[attr]
        number_of_vals = get_number_of_vals(attr_arr)
        two_sensible_classes_all[attr] = number_of_vals > 1

    return two_sensible_classes_all


def check_binary(arr):
    """
     This function is used from arXiv:1507.05259 [stat.ML] source code. Source link below.

    ----------
    Parameters

    param arr: ndarray, shape (n_samples,).
    return: bool. True if every entry of the array is 0 or 1.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    s = sorted(set(arr))
    if s[0] == 0 and s[1] == 1:
        return True
    raise Exception('Sensitive attribute not binary')




def get_number_of_vals(arr):
    """
    Parameters

    param arr: ndarray, shape (n_samples,).
    return: int. Number of unique values in an array.
    """

    return len(set(arr))


# %%
# Data encoding

def encode_data(df, protected_attribute, protected_group_name, class_label, advantageous_label_value, continuous):
    df_enc = df.copy()

    # Sensitive encoding
    df_enc[protected_attribute] = [0 if v == protected_group_name else 1 for v in df[protected_attribute]]

    # Output class encoding
    df_enc[class_label] = [1 if v == advantageous_label_value else -1 for v in df[class_label]]

    # Replace non-numeric variables with one-hot encoding
    df_enc, numeric_attrs_idx_set = get_one_hot_encoding(df_enc, continuous)

    return df_enc, numeric_attrs_idx_set


def get_one_hot_encoding(df, continuous):
    if continuous:
        numeric_attrs_idx_set = None
    else:
        numeric_attrs_idx_set = set()
    do_print = False
    df_one_hot = df.copy()
    columns = df.columns
    i = 0
    le = preprocessing.LabelEncoder()
    for c in columns:
        df_one_hot.reset_index(drop=True, inplace=True)
        if df[c].dtypes == 'object':
            n_unique = df[c].nunique()
            if n_unique > 2:
                df_dummies = pd.get_dummies(df[c])
                df_dummies.reset_index(drop=True, inplace=True)
                n_dummies = df_dummies.shape[1]
                assert(n_dummies == n_unique)
                if do_print:
                    print('  %s: categorical, n.cats: %d' % (c, n_dummies))
                dummies_names = ['%s %d' % (c, e) for e in range(n_dummies)]
                df_dummies.columns = dummies_names
                df_one_hot = pd.concat([df_one_hot.iloc[:, :i], df_dummies], axis=1)
                i += len(df_dummies.columns)
            else:
                if do_print:
                    print('  %s: binary, n.vals: %d' % (c, n_unique))
                masked_col = pd.DataFrame({c: le.fit_transform(df[c])})
                masked_col.reset_index(drop=True, inplace=True)
                df_one_hot = pd.concat([df_one_hot.iloc[:, :i], masked_col], axis=1)
                i += 1
            if do_print:
                print('   unique values:', df[c].unique())
        else:
            if do_print:
                print('  %s: numerical, range:[%0.2f, %0.2f]' % (c, min(df[c]), max(df[c])))
            df_col = pd.DataFrame({c: np.array(df[c])})
            df_one_hot = pd.concat([df_one_hot.iloc[:, :i], df_col], axis=1)
            if numeric_attrs_idx_set is not None:
                numeric_attrs_idx_set.add(i)
            i += 1
    return df_one_hot, numeric_attrs_idx_set


#%%
def divide_df(df, protected_attribute, class_label):

    # Divide data
    df_temp = df.copy()
    a = df_temp.pop(protected_attribute).to_numpy(dtype=np.float64)
    y = df_temp.pop(class_label).to_numpy(dtype=np.float64)
    X = df_temp.to_numpy(dtype=np.float64)

    return X, y, {protected_attribute: a}


def shuffle_data(X, a_dict, y, m_seed):
    # shuffle data
    seed(m_seed)
    np.random.seed(m_seed)
    perm = list(range(len(y)))
    shuffle(perm)
    X_new = X[perm]
    y_new = y[perm]
    a_dict_new = dict()
    for k in a_dict.keys():
        a_dict_new[k] = a_dict[k][perm]
    return X_new, y_new, a_dict_new


def crop_data(X, a_dict, y, n_sample):
    # Return only n_sample examples from the data
    X_new = X[:n_sample]
    y_new = y[:n_sample]
    a_dict_new = dict()
    for k in a_dict.keys():
        a_dict_new[k] = a_dict[k][:n_sample]
    return X_new, y_new, a_dict_new


def get_sample(X, y, a_dict, m_seed, n_sample=-1, do_shuffle=True):

    if do_shuffle:
        X, y, a_dict = shuffle_data(X, a_dict, y, m_seed)
    if n_sample != -1 and n_sample is not None:
        X, y, a_dict = crop_data(X, a_dict, y, n_sample)

    return X, y, a_dict


def split_data(X, a, y, protected_attribute, val_size=0.2, test_size=0.2):
    # Splitting data into train, test and validation
    a = a[protected_attribute]
    X_train, X_test, a_train, a_test, y_train, y_test = train_test_split(X, a, y, test_size=test_size, shuffle=False)

    if val_size is None:
        X_val, a_val, y_val = None, None, None
    else:
        val_size_temp = val_size / (1. - test_size)
        X_train, X_val, a_train, a_val, y_train, y_val = train_test_split(X_train, a_train, y_train,
                                                                          test_size=val_size_temp, shuffle=False)

    return X_train, X_val, X_test, {protected_attribute: a_train}, {protected_attribute: a_val}, \
           {protected_attribute: a_test}, y_train, y_val, y_test


def shuffle_and_split(x, y, sensitive,  s_attr, train_fold_size, m_seed, n_sample, do_shuffle=True):

    sensitive = sensitive[s_attr]

    train_size = int(n_sample * train_fold_size)
    test_size = int(n_sample * 1.-train_fold_size)

    x_train, x_test, a_train, a_test, y_train, y_test = train_test_split(x, sensitive, y, test_size=test_size,
                                                                         train_size=train_size, random_state=m_seed,
                                                                         shuffle=do_shuffle)

    a_test, a_train = {s_attr: a_test}, {s_attr: a_train}
    return x_train, y_train, a_train, x_test, y_test, a_test