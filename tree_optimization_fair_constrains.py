# coding=utf-8
# %%
from scipy.optimize import minimize
from data_utils import check_binary
import numpy as np
from evaluations_accuracy import get_distance_to_split


# %%


def get_sensitive_attr_cov(model, x_arr, dist_boundary_arr, x_sensitive):
    """

    This function is adapted from arXfiv:1507.05259 [stat.ML] source code below.

    ----------
    Parameters

    param model: coefficients determining a model of decision boundary with intercept.
    param x_arr: ndarray, shape (n_samples, n_features+1). Training data plus a colum of intercept.
    param dist_boundary_arr: ndarray, shape (n_samples,). Distances of samples to decision boundary.
    param x_sensitive: ndarray, shape (n_samples,). Sensitive attribute of samples.
    param verbose: bool. For reporting.
    return: float. Covariance between the sensitive attribute value and the distance from the decison boundary.
    -------
    Notes

    If the model is None, we assume that the dist_boundary_arr contains the distace from the decision boundary.
    If the model is not None, we just compute a dot product or model and x_arr.
    For the case of SVM and Decision Tree, we pass the distace from boundary becase the intercept is internalized
    for the class, and we have computed the distance using the project function.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    if x_sensitive.ndim != 1:
        raise Exception('Sensitive arr should have one column')

    distances_boundary = get_distance_to_split(x=x_arr, pred_idx=1, w=model, arr=dist_boundary_arr)

    assert (x_arr.shape[0] == x_sensitive.shape[0])
    cov = np.dot(x_sensitive - np.mean(x_sensitive), distances_boundary) / float(len(x_sensitive))

    return cov


def test_sensitive_attr_constraint_cov(model, x_arr, dist_boundary_arr, x_sensitive, thresh):
    """
    This function is used from arXiv:1507.05259 [stat.ML] source code below.

    ----------
    Parameters

    param model: coefficients determining a model of decision boundary with intercept.
    param x_arr: ndarray, shape (n_samples, n_features+1). Training data plus a colum of intercept.
    param dist_boundary_arr: ndarray, shape (n_samples,). Distances of samples to decision boundary.
    param x_sensitive: ndarray, shape (n_samples,). Sensitive attribute of samples.
    param thresh: float. Positive value determining a fairness constrain.
    param verbose: bool. For reporting.
    return: float. Value is negative if the fair constraint is not satisfied, is positive otherwise.
    The constraint is of the form: |cov| < thresh . Were the covariance is calculated between
    the sensitive attribute value and the distance from the decison boundary.
    -------
    Notes

    If the model is None, we assume that the dist_boundary_arr contains the distace from the decision boundary.
    If the model is not None, we just compute a dot product or model and x_arr.
    For the case of SVM, we pass the distace from boundary becase the intercept is internalized for the class,
    and we have computed the distance using the project function.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    assert (thresh >= 0.)

    distances_boundary = get_distance_to_split(x=x_arr, pred_idx=1, w=model, arr=dist_boundary_arr)

    cov = get_sensitive_attr_cov(model=model, x_arr=x_arr, dist_boundary_arr=distances_boundary,
                                 x_sensitive=x_sensitive)

    ans = thresh - abs(cov)

    return ans


def check_one_hot_encoding(in_arr):
    """
    This function is used from arXiv:1507.05259 [stat.ML] source code. Source link below.

    ----------
    Parameters

    param in_arr: 1-D array with int vals; representing a discrete variable.
    return: if the discrete variable is binary returns the same array. Else, returns:
     m (ndarray) one-hot encoded matrix (one binary column for each category of the discrete variable),
     d (dict) also returns a dictionary original_val -> column in encoded matrix.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    assert (in_arr.ndim == 1)

    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if num_uniq_vals == 1:
        return None, None

    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    raise Exception('Sensitive attribute not binary')


def get_constraint_list_cov(x_train, y_train, x_sensitive_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):
    """
    This function is adapted from arXiv:1507.05259 [stat.ML] source code below.

    ----------
    Parameters

    param x_train: ndarray, shape (n_samples, n_features+1). Training data plus a colum of intercept.
    param y_train:ndarray, shape (n_samples,). Training labels.
    param x_sensitive_train: ndarray, shape (n_samples,). Sensitive attribute of samples.
    param sensitive_attrs: list of strings. Names of sensitive attributes.
    param sensitive_attrs_to_cov_thresh: dict. Key = name of sensitive attribute, value = thresh determinig
    the fair restriction.
    return: list of dicts. One dict per binary (possible dummy) attribute. Dictionaires needed for SLSQP scipy 
    implementation are of the form: c = {'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, 
    y_train, attr_arr_transformed, thresh, False)}

    -------
    Notes

    If the model is None, we assume that the dist_boundary_arr contains the distace from the decision boundary.
    If the model is not None, we just compute a dot product or model and x_arr.
    For the case of SVM, we pass the distace from boundary becase the intercept is internalized for the class,
    and we have computed the distance using the project function.

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    constraints = []

    for attr in sensitive_attrs:

        attr_arr = x_sensitive_train[attr]

        attr_arr_transformed, index_dict = check_one_hot_encoding(attr_arr)

        # if there is only one class in the sensitive attribute, the fair restrictions are removed
        # this case arrise sometimes when optimization is used in decision trees
        # TODO: Ahorita sólo funciona para un sesitive attribute, extender para multivariado
        if attr_arr_transformed is None:
            return None

        # binary attribute por qué y_train donde tengo dist_boundary_arr
        if index_dict is None:
            thresh = sensitive_attrs_to_cov_thresh[attr]
            # si test_sensitive_attr_constraint_cov(model, x_arr, dist_boundary_arr, x_sensitive, thresh)
            # en donde está None, tenían y_train:
            c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                  'args': (x_train, None, attr_arr_transformed, thresh)})
            constraints.append(c)

    return constraints


def train_model(x, y, x_sensitive, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
                sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None):
    """
    This function is adapted from arXiv:1507.05259 [stat.ML] source code below.

    It trains the model subject to various fairness or accuracy constraints.
    If no constraints are given, then simply trains an unaltered classifier.

    ----------
    Parameters

    param x: ndarray, shape (n_samples, n_features+1). Training data plus a colum of intercept.
    param y: ndarray, shape (n_samples,). Training labels.
    param x_sensitive: dict. Key = string name of sensitive attribute, value = ndarray, shape (n_samples,),
        the values sensitive attribute of training samples.
    param loss_function: the loss function that we want to optimize. For now we have implementation of logistic loss.
    TODO: convex??
    param apply_fairness_constraints: int. Optimize accuracy subject to fairness constraint (0/1 values)
    param apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    param sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
    param sensitive_attrs: list of strings. Names of sensitive attributes for which to apply fairness constraint.
         All of these sensitive features should have a corresponding array in x_sensitive.
    param sensitive_attrs_to_cov_thresh: dict. Key = name of sensitive attribute, value = threshold determinig
    the fair restriction (this is only needed when apply_fairness_constraints=1)
    param gamma: float. Value in (0,1). Controls the loss in accuracy we are willing to incur when
    using apply_accuracy_constraint and sep_constraint.
    return: w, ndarray shape (n_coefficiente,). The learned weight vector for the classifier.

    -------

    Notes

    Both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time.
    If the model is None, we assume that the dist_boundary_arr contains the distace from the decision boundary.
    If the model is not None, we just compute a dot product or model and x_arr.
    For the case of SVM, we pass the distace from boundary becase the intercept is internalized for the class,
    and we have computed the distance using the project function.

    -----

    Source code at:

    https://github.com/mbilalzafar/fair-classification.git


    """

    # both constraints cannot be applied at the same time
    assert (not (apply_accuracy_constraint == 1 and apply_fairness_constraints == 1))
    assert (x.shape[1] == 2)
    # era 100000 en el original
    max_iter = 3000  # maximum number of iterations for the minimization algorithm

    one_sensible_class = False  # notify if there is only one class in the sensitive attribute
    # in that case, the fair restrictions are removed, arrises sometimes when optimization is used in decision trees

    # initial point
    cut_0 = (np.min(x[:, 1]) + np.max(x[:, 1])) / 2.
    w_1 = 1. / (cut_0 * cut_0 + 1.)
    w_0 = np.sqrt(1. - w_1 * w_1)
    x_0 = np.array([w_0, w_1])

    if apply_fairness_constraints == 0:
        constraints = []
    else:
        # list of dictionaries of optimization constraints
        constraints = get_constraint_list_cov(x, y, x_sensitive, sensitive_attrs, sensitive_attrs_to_cov_thresh)
        if constraints is None:  # if there are just one class in sensitive attribute, remove fair constraints
            one_sensible_class = True
            constraints = []

    # train without restrictions (unconstrained)
    w_u = minimize(fun=loss_function,
                 # x0=np.random.rand(x.shape[1], ),
                 x0=x_0,
                 args=(x, y),  # tuple of the fixed parameters needed to completely specify the loss function
                 method='SLSQP',  # Sequential Least Squares Programming
                 options={"maxiter": max_iter},
                 constraints=[]
                 )

    # train with cross cov constraints
    w = minimize(fun=loss_function,
                 x0=x_0,
                 args=(x, y),  # tuple of the fixed parameters needed to completely specify the loss function
                 method='SLSQP',  # Sequential Least Squares Programming
                 options={"maxiter": max_iter},
                 constraints=constraints
                 )

    return w.x, one_sensible_class, w_u.x


def get_all_covariance_sensitive_attrs(model, x_arr, dist_boundary_arr, x_sensitive, sensitive_attrs, get_dict=True):
    """
    This function is adapted from arXiv:1507.05259 [stat.ML] source code below.

    ----------
    Parameters

    param model: coefficients determining a model of decision boundary with intercept.
    param x_arr: ndarray, shape (n_samples, n_features+1). Training data plus a colum of intercept.
    param dist_boundary_arr: ndarray, shape (n_samples,). Distances of samples to decision boundary.
    param x_sensitive: ndarray, shape (n_samples,). Sensitive attribute of samples.
    param sensitive_attrs: list of strings. Names of sensitive attributes.
    return: dictionary key=sensitive attribute, value = 2 cases:
     - case binary sensitive attribute:
        value = absolute of the covariance between sensitive attribute and distance from decision boundary
     - case the sensitive attribute has more than 2 categorical values:
        value = dictionary key=category, value=absolute covariance between dummy sensitive attribute and
        distance from decision boundary

    -----
    Source code at:

    https://github.com/mbilalzafar/fair-classification.git

    """

    distances_boundary = get_distance_to_split(x=x_arr, pred_idx=1, w=model, arr=dist_boundary_arr)

    if get_dict:
        sensitive_attrs_to_cov = dict()
    else:
        sensitive_attrs_to_cov = []

    for attr_name in sensitive_attrs:

        attr_arr = x_sensitive[attr_name]
        check_binary(attr_arr)

        cov = get_sensitive_attr_cov(model=None, x_arr=x_arr, dist_boundary_arr=distances_boundary,
                                     x_sensitive=attr_arr)

        if get_dict:
            sensitive_attrs_to_cov[attr_name] = cov
        else:
            sensitive_attrs_to_cov.append(cov)

    return sensitive_attrs_to_cov
