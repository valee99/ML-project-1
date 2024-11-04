"""Functions to find the best model with a grid search looping on several hyperparameters and reporting the k-fold cross validation F1-Score for each combination."""

import numpy as np
import time
import json

from preprocessing import *
from feature_eng import *
from implementations import *
from metrics import *
from helpers import *


def build_k_indices(y: np.array, k_fold: int, seed: int) -> np.array:
    """Creates an array with the indices of the datapoints belonging to each fold of the cross-validation

    Args:
        y: shape=(N, )
        k_fold: a scalar denoting the number of folds for the cross-validation
        seed: a scalar setting the seed for the random operations

    Returns:
        k_indices: shape=(k_fold, interval)
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)    # randomly shuffle indices
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_logreg_thresh(
    x_tr: np.array,
    y_tr: np.array,
    x_te: np.array,
    y_te: np.array,
    features_list_train: list[str],
    lambda_: float,
    gamma: float,
    alpha: float,
    max_iters: int,
    weight: dict[float],
    oversampling_ratio: float,
    batch_size: int,
    thresholds: list[float],
    cluster: bool,
    max_degree: int,
    num_list: list[str],
    seed: int,
) -> tuple[dict[float], np.array]:
    """Runs a logistic regression with the provided parameters and training dataset and returns the metrics for a list of thresholds on the test dataset

    Args:
        x_tr: shape=((k-1)*interval, D)
        y_tr: shape=((k-1)*interval, )
        x_te: shape=(interval, D)
        y_te: shape=(interval, )
        features_list_train: the list of features in the dataset in the same order as the columns
        lambda_: a scalar denoting the strength of the regularization term
        gamma: a scalar denoting the strength of the learning rate
        alpha: a scalar denoting the balance between the Ridge and Lasso regularization
        max_iters: a scalar denoting the number of iterations to run
        weight: a dictionary with the weights of the errors of each class
        oversampling_ratio: a scalar denoting the target ratio of the positive class to reach by oversampling with replacement
        batch_size: a scalar denoting the size of the batch at each iteration for the SGD
        thresholds: a list of possible thresholds for prediction
        cluster: a boolean denoting if a feature based on a k-mean clustering is added
        max_degree: a scalar denoting the maximum degree of the polynomial expansion on numerical features
        num_list: a list with the names of the numerical features
        seed: a scalar setting the seed for the random operations

    Returns:
        f1_score_dict: a dictionary with the thresholds as keys and the F1-Scores as values
        accuracy_dict: a dictionary with the thresholds as keys and the accuracies as values
        time_dict: a dictionary with the thresholds as keys and the times as values
        w: optimal weights, numpy array of shape(D, ), D is the number of features.
    """

    np.random.seed(seed)
    # set class weights if weight parameter is true
    if weight == True:
        class_weight = {
            0: len(y_tr) / (2 * np.bincount(y_tr)[0]),  # For class 0 (negative)
            1: len(y_tr) / (2 * np.bincount(y_tr)[1]),  # For class 1 (positive)
        }
    else:
        class_weight = {0: 1, 1: 1}

    # polynomial expansion
    if max_degree != 1:
        for feature in num_list:
            if feature in features_list_train:
                x_tr = polynomial_expansion(
                    x_tr, feature, features_list_train, max_degree
                )
                x_te = polynomial_expansion(
                    x_te, feature, features_list_train, max_degree
                )
    
    # add a k-means clustering feature
    if cluster:
        centroids, labels = kmeans(x_tr, 2, 12)
        x_tr = np.concatenate((x_tr, labels[:, np.newaxis]), axis=1)
        distances_te = compute_distances(x_te, centroids)
        labels_te = np.argmin(distances_te, axis=1)
        x_te = np.concatenate((x_te, labels_te[:, np.newaxis]), axis=1)

    # oversample positive class
    if oversampling_ratio > 0:

        n_posi = (y_tr == 1).sum()
        n_nega = (y_tr == 0).sum()
        y_tr_posi_idx = np.argwhere(y_tr == 1).flatten()
        
        # calculate number of positive samples needed for the target ratio
        n_posi_to_sample = int(
            (oversampling_ratio * (n_posi + n_nega) - n_posi) / (1 - oversampling_ratio)
        )
        posi_sample_idx = np.random.choice(
            y_tr_posi_idx, n_posi_to_sample, replace=True
        )

        x_tr_sample = x_tr[posi_sample_idx]
        y_tr_sample = y_tr[posi_sample_idx]

        # add to the trainning dataset
        x_tr = np.concatenate((x_tr, x_tr_sample), axis=0)
        y_tr = np.concatenate((y_tr, y_tr_sample), axis=0)

    # initialize weights for training
    initial_w = np.zeros(x_tr.shape[1], dtype=float)

    start_time = time.time()
    w, _ = elastic_net_logistic_regression_weighted(
        y_tr,
        x_tr,
        lambda_,
        initial_w,
        max_iters,
        gamma,
        class_weight,
        alpha,
        batch_size,
    )
    end_time = time.time()

    # evaluate model at different threshold values
    f1_score_dict = dict()
    accuracy_dict = dict()
    time_dict = dict()
    for threshold in thresholds:
        y_pred_te = np.where(np.dot(x_te, w) >= threshold, 1, -1)
        y_te_converted = np.where(y_te == 0, -1, y_te)
        f1_score_dict[str(threshold)] = compute_f1_score(y_te_converted, y_pred_te)
        accuracy_dict[str(threshold)] = compute_accuracy(y_te_converted, y_pred_te)
        time_dict[str(threshold)] = end_time - start_time
    return f1_score_dict, accuracy_dict, time_dict, w


def grid_search_cv_to_json(
    y: np.array,
    x: np.array,
    parameters: dict,
    num_list: list[str],
    values_null_list: list[list[int]],
    columns_null_list: list[list[str]],
    features_list: list[str],
    binary_list: list[str],
    binary_reverse_list: list[str],
    categorical_label_list: list[str],
    reverse_categorical_label_list: list[str],
    one_hot_encoding_list: list[str],
):
    """Runs a grid search on the provided list of parameters and save the resulting metrics (mean and standard deviation) obtained on the validation fold in a JSON file to select the best model afterwards

    Args:
        y: shape=(N, )
        x: shape=(N, D)
        parameters: a dictionnary with the lists of parameters to loop on in the grid search
        num_list: the list of the names of the numerical features to process
        values_null_list: a list of the lists of specific values to replace
        columns_null_list: a list of the lists of features corresponding to the specific values to replace
        features_list: the list of features in the dataset in the same order as the columns
        binary_list: the list of the names of the binary features to process
        binary_reverse_list: the list of the names of the reverse binary features to process
        categorical_label_list: the list of the names of the categorical features to process
        reverse_categorical_label_list: the list of the names of the reverse categorical features to process
        one_hot_encoding_list: the list of the names of the categorical features to process with one-hot encoding
    """

    y_converted = np.where(y == -1, 0, y)
    seed = 12

    k_fold = parameters["k_fold"]
    oversampling_ratio = parameters["oversampling_ratio"]
    weights = parameters["use_weight"]
    lambdas = parameters["lambdas"]
    gammas = parameters["gammas"]
    alphas = parameters["alphas"]
    max_iters = parameters["max_iters"]
    batch_sizes = parameters["batch_size"]
    thresholds = parameters["threshold"]
    max_degrees = parameters["max_degree"]
    clusters = parameters["clusters"]

    # k-fold indices
    k_indices = build_k_indices(y_converted, k_fold, seed)

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    features_list_trains = []

    for k in range(k_fold):
        
        # define training and test indices for the k-th fold
        tr_indices = k_indices[
            [i for i in range(k_indices.shape[0]) if i != k]
        ].flatten()
        te_indices = k_indices[k]
        x_tr = x[tr_indices]
        y_tr = y_converted[tr_indices]
        x_te = x[te_indices]
        y_te = y_converted[te_indices]

        features_list_train = features_list.copy()
        features_list_test = features_list.copy()

        # pre-processing
        x_tr = preprocess_data(
            x_tr,
            values_null_list,
            columns_null_list,
            features_list_train,
            num_list,
            binary_list,
            binary_reverse_list,
            categorical_label_list,
            reverse_categorical_label_list,
            one_hot_encoding_list,
        )
        x_te = preprocess_data(
            x_te,
            values_null_list,
            columns_null_list,
            features_list_test,
            num_list,
            binary_list,
            binary_reverse_list,
            categorical_label_list,
            reverse_categorical_label_list,
            one_hot_encoding_list,
        )

        x_trains.append(x_tr)
        y_trains.append(y_tr)
        x_tests.append(x_te)
        y_tests.append(y_te)
        features_list_trains.append(features_list_train)

    parameters_combinations = []

    for gamma in gammas:

        for lambda_ in lambdas:

            for alpha in alphas:

                if lambda_ == 0 and alpha != 0:
                    continue

                for ratio in oversampling_ratio:

                    for weight in weights:

                        for batch_size in batch_sizes:

                            for max_degree in max_degrees:

                                for cluster in clusters:

                                    for max_iter in max_iters:
                                        # store metrics for each fold
                                        f1_score_max_iter = []
                                        accuracy_max_iter = []
                                        time_max_iter = []
                                        
                                        # k-fold cross validation
                                        for k in range(k_fold):
                                            f1_score, accuracy, time_dict, _ = (
                                                cross_validation_logreg_thresh(
                                                    x_trains[k],
                                                    y_trains[k],
                                                    x_tests[k],
                                                    y_tests[k],
                                                    features_list_trains[k],
                                                    lambda_,
                                                    gamma,
                                                    alpha,
                                                    max_iter,
                                                    weight,
                                                    ratio,
                                                    batch_size,
                                                    thresholds,
                                                    cluster,
                                                    max_degree,
                                                    num_list,
                                                    seed,
                                                )
                                            )
                                            f1_score_max_iter.append(f1_score)
                                            accuracy_max_iter.append(accuracy)
                                            time_max_iter.append(time_dict)

                                        # mean and std for each threshold
                                        for threshold in thresholds:
                                            f1_score_mean = np.mean(
                                                [
                                                    f1_score_max_iter[i][str(threshold)]
                                                    for i in range(
                                                        len(f1_score_max_iter)
                                                    )
                                                ]
                                            )
                                            accuracy_mean = np.mean(
                                                [
                                                    accuracy_max_iter[i][str(threshold)]
                                                    for i in range(
                                                        len(accuracy_max_iter)
                                                    )
                                                ]
                                            )
                                            time_mean = np.mean(
                                                [
                                                    time_max_iter[i][str(threshold)]
                                                    for i in range(len(time_max_iter))
                                                ]
                                            )
                                            f1_score_std = np.std(
                                                [
                                                    f1_score_max_iter[i][str(threshold)]
                                                    for i in range(
                                                        len(f1_score_max_iter)
                                                    )
                                                ]
                                            )
                                            accuracy_std = np.std(
                                                [
                                                    accuracy_max_iter[i][str(threshold)]
                                                    for i in range(
                                                        len(accuracy_max_iter)
                                                    )
                                                ]
                                            )
                                            time_std = np.std(
                                                [
                                                    time_max_iter[i][str(threshold)]
                                                    for i in range(len(time_max_iter))
                                                ]
                                            )

                                            # store parameter combination and results
                                            parameters_combinations.append(
                                                {
                                                    "gamma": gamma,
                                                    "lambda": lambda_,
                                                    "alpha": alpha,
                                                    "ratio": ratio,
                                                    "weight": weight,
                                                    "max_iter": max_iter,
                                                    "threshold": threshold,
                                                    "batch_size": batch_size,
                                                    "cluster": cluster,
                                                    "max_degree": max_degree,
                                                    "time_mean": time_mean,
                                                    "time_std": time_std,
                                                    "f1_score_mean": f1_score_mean,
                                                    "f1_score_std": f1_score_std,
                                                    "accuracy_mean": accuracy_mean,
                                                    "accuracy_std": accuracy_std,
                                                }
                                            )
                                            print(
                                                {
                                                    "gamma": gamma,
                                                    "lambda": lambda_,
                                                    "alpha": alpha,
                                                    "ratio": ratio,
                                                    "weight": weight,
                                                    "max_iter": max_iter,
                                                    "threshold": threshold,
                                                    "batch_size": batch_size,
                                                    "cluster": cluster,
                                                    "max_degree": max_degree,
                                                    "time_mean": time_mean,
                                                    "time_std": time_std,
                                                    "f1_score_mean": f1_score_mean,
                                                    "f1_score_std": f1_score_std,
                                                    "accuracy_mean": accuracy_mean,
                                                    "accuracy_std": accuracy_std,
                                                }
                                            )

    with open("results_per_parameters_sgd.json", "w") as results_file:
        json.dump(parameters_combinations, results_file)


if __name__ == "__main__":

    # the lists follow the same structure as in nan_and_cross_correlation_analysis.ipynb
    BINARY_LIST = [
        "SEX",
        "_RFHLTH",
        "_HCVU651",
        "_RFHYPE5",
        "_RFCHOL",
        "_LTASTH1",
        "_CASTHM1",
        "_RACEG21",
        "_AGE65YR",
        "_RFBMI5",
        "_RFSMOK3",
        "_RFBING5",
        "_RFDRHV5",
        "_FRTLT1",
        "_VEGLT1",
        "_FRUITEX",
        "_VEGETEX",
        "_TOTINDA",
        "PAMISS1_",
        "_PAINDX1",
        "_PASTRNG",
        "_PASTAE1",
        "_RFSEAT2",
        "_RFSEAT3",
        "_FLSHOT6",
        "_PNEUMO2",
        "_AIDTST3",
    ]
    BINARY_REVERSE_LIST = [
        "_DRDXAR1",
        "_HISPANC",
        "DRNKANY5",
        "_FRTRESP",
        "_VEGRESP",
        "_FRT16",
        "_VEG23",
        "RDUCHART",
        "DRADVISE",
        "ADANXEV",
    ]
    ONE_HOT_ENCODING_LIST = [
        "_CHOLCHK",
        "_ASTHMS1",
        "_PRACE1",
        "_MRACE1",
        "_RACE",
        "_RACEGR3",
        "_RACE_G1",
        "_AGEG5YR",
        "_AGE80",
        "_SMOKER3",
        "_PAREC1",
        "_LMTACT1",
        "_LMTWRK1",
        "_LMTSCL1",
    ]
    CATEGORICAL_LABEL_LIST = [
        "_AGE_G",
        "_BMI5CAT",
        "_CHLDCNT",
        "_EDUCAG",
        "_INCOMG",
        "ACTIN11_",
        "ACTIN21_",
    ]
    REVERSE_CATEGORICAL_LABEL_LIST = ["_PACAT1", "_PA150R2", "_PA300R2", "_PA30021"]
    NUM_LIST = [
        "HTIN4",
        "HTM4",
        "WTKG3",
        "_BMI5",
        "DROCDY3_",
        "_DRNKWEK",
        "FTJUDA1_",
        "FRUTDA1_",
        "BEANDAY_",
        "GRENDAY_",
        "ORNGDAY_",
        "VEGEDA1_",
        "_MISFRTN",
        "_MISVEGN",
        "_FRUTSUM",
        "_VEGESUM",
        "METVL11_",
        "METVL21_",
        "MAXVO2_",
        "FC60_",
        "PADUR1_",
        "PADUR2_",
        "PAFREQ1_",
        "PAFREQ2_",
        "_MINAC11",
        "_MINAC21",
        "STRFREQ_",
        "PAMIN11_",
        "PAMIN21_",
        "PA1MIN_",
        "PAVIG11_",
        "PAVIG21_",
        "PA1VIGM_",
    ]

    NULL_9 = [
        "_RFHLTH",
        "_HCVU651",
        "_RFHYPE5",
        "_CHOLCHK",
        "_RFCHOL",
        "_LTASTH1",
        "_CASTHM1",
        "_ASTHMS1",
        "_HISPANC",
        "_RACE",
        "_RACEG21",
        "_RACEGR3",
        "_RFBMI5",
        "_CHLDCNT",
        "_EDUCAG",
        "_INCOMG",
        "_SMOKER3",
        "_RFSMOK3",
        "_RFBING5",
        "_RFDRHV5",
        "_FRTLT1",
        "_VEGLT1",
        "_TOTINDA",
        "PAMISS1_",
        "_PACAT1",
        "_PAINDX1",
        "_PA150R2",
        "_PA300R2",
        "_PA30021",
        "_PASTRNG",
        "_PAREC1",
        "_PASTAE1",
        "_LMTACT1",
        "_LMTWRK1",
        "_LMTSCL1",
        "_RFSEAT2",
        "_RFSEAT3",
        "_FLSHOT6",
        "_PNEUMO2",
        "_AIDTST3",
    ]
    NULL_77_99 = ["_PRACE1", "_MRACE1"]
    NULL_14 = ["_AGEG5YR"]
    NULL_3 = ["_AGE65YR"]
    NULL_99999 = ["WTKG3"]
    NULL_99900 = ["_DRNKWEK", "MAXVO2_", "FC60_", "PAFREQ1_", "PAFREQ2_"]
    NILL_99000 = ["STRFREQ_"]
    NULL_7_9 = ["DRNKANY5", "DIABETE3", "HAREHAB1", "STREHAB1", "ADANXEV", "DRADVISE"]
    NULL_900 = ["DROCDY3_"]

    COLUMNS_NULL_LIST = [
        NULL_9,
        NULL_77_99,
        NULL_14,
        NULL_3,
        NULL_99999,
        NULL_99900,
        NILL_99000,
        NULL_7_9,
        NULL_900,
    ]
    VALUES_NULL_LIST = [
        [9],
        [77, 99],
        [14],
        [3],
        [99999],
        [99900],
        [99000],
        [7, 9],
        [900],
    ]

    print("Loading Dataset...")
    dataset = load_csv_data("./dataset")
    with open("./dataset/x_test.csv") as file:
        HEADER = file.readline().strip().split(",")[1:]
    print("Dataset Loaded !")

    # from nan_and_cross_correlation_analysis.ipynb
    final_features = [
        "_RACE_G1",
        "DIABETE3",
        "SEX",
        "_RFHLTH",
        "_RFHYPE5",
        "_CHOLCHK",
        "_RFCHOL",
        "_ASTHMS1",
        "_DRDXAR1",
        "_AGE_G",
        "HTM4",
        "_BMI5CAT",
        "_INCOMG",
        "_RFSMOK3",
        "DRNKANY5",
        "DROCDY3_",
        "_RFBING5",
        "_DRNKWEK",
        "_RFDRHV5",
        "_FRUTSUM",
        "_VEGESUM",
        "_FRTLT1",
        "_VEGLT1",
        "_VEGETEX",
        "FC60_",
        "STRFREQ_",
        "PAMISS1_",
        "_PACAT1",
        "_PASTRNG",
        "_PAREC1",
        "_RFSEAT3",
        "_FLSHOT6",
        "_PNEUMO2",
        "_AIDTST3",
    ]

    x_train, x_test, y_train, _, _ = dataset
    x_train_selected = select_features(x_train, final_features, HEADER)

    # FULL_PARAMETERS = {"k_fold":5,"oversampling_ratio":[0,0.3,0.4,0.5],"use_weight":[False,True],"lambdas":[0] + [5*(10**i) for i in range(-4,3)],"gammas":[0] +[5*(10**i) for i in range(-3,3)],"alphas":[0,0.25,0.5,0.75,1],"max_iters":[100,500,1000,2000,3000,4000],"batch_size":[0,len(x_train)//2,len(x_train)//4,len(x_train)//10],"threshold":[0.2,0.3,0.4,0.5],"max_degree":[1,2,3],"clusters":[False,True]}
    FULL_PARAMETERS = {
        "k_fold": 5,
        "oversampling_ratio": [0.4],
        "use_weight": [True],
        "lambdas": [0] + [5 * (10**i) for i in range(-4, 3)],
        "gammas": [5 * (10**i) for i in range(-3, 3)],
        "alphas": [0, 0.25, 0.5, 0.75, 1],
        "max_iters": [500],
        "batch_size": [len(x_train) // 10],
        "threshold": [0.2, 0.3, 0.4, 0.5],
        "max_degree": [1, 2, 3],
        "clusters": [False, True],
    }

    grid_search_cv_to_json(
        y_train,
        x_train_selected,
        FULL_PARAMETERS,
        NUM_LIST,
        VALUES_NULL_LIST,
        COLUMNS_NULL_LIST,
        final_features,
        BINARY_LIST,
        BINARY_REVERSE_LIST,
        CATEGORICAL_LABEL_LIST,
        REVERSE_CATEGORICAL_LABEL_LIST,
        ONE_HOT_ENCODING_LIST,
    )
