"""Script to make the predictions."""

import numpy as np

from preprocessing import *
from feature_eng import *
from implementations import *
from metrics import *
from helpers import *

if __name__ == "__main__":

    SEED = 12

    np.random.seed(SEED)

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

    PARAMETERS = {
        "k_fold": 5,
        "oversampling_ratio": 0.4,
        "use_weight": False,
        "lambda": 0,
        "gamma": 0.05,
        "alpha": 0,
        "max_iter": 10000,
        "batch_size": 0,
        "threshold": 0.46,
        "max_degree": 1,
        "cluster": True,
    }

    dataset = load_csv_data("dataset")
    with open("./dataset/x_test.csv") as file:
        HEADER = file.readline().strip().split(",")[1:]

    final_features = [
        "HAREHAB1",
        "DRADVISE",
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

    x_train, x_test, y_train, train_ids, test_ids = dataset
    x_train_selected = select_features(x_train, final_features, HEADER)
    x_test_selected = select_features(x_test, final_features, HEADER)

    features_list_train = final_features.copy()
    features_list_test = final_features.copy()

    x_train_processed, features_list_train = preprocess_data(
        x_train_selected,
        VALUES_NULL_LIST,
        COLUMNS_NULL_LIST,
        features_list_train,
        NUM_LIST,
        BINARY_LIST,
        BINARY_REVERSE_LIST,
        CATEGORICAL_LABEL_LIST,
        REVERSE_CATEGORICAL_LABEL_LIST,
        ONE_HOT_ENCODING_LIST,
    )
    x_test_processed, features_list_test = preprocess_data(
        x_test_selected,
        VALUES_NULL_LIST,
        COLUMNS_NULL_LIST,
        features_list_test,
        NUM_LIST,
        BINARY_LIST,
        BINARY_REVERSE_LIST,
        CATEGORICAL_LABEL_LIST,
        REVERSE_CATEGORICAL_LABEL_LIST,
        ONE_HOT_ENCODING_LIST,
    )

    y_train_converted = np.where(y_train == -1, 0, y_train)

    if PARAMETERS["use_weight"] == True:
        class_weight = {
            0: len(y_train_converted)
            / (2 * np.bincount(y_train_converted)[0]),  # For class 0 (negative)
            1: len(y_train_converted)
            / (2 * np.bincount(y_train_converted)[1]),  # For class 1 (positive)
        }
    else:
        class_weight = {0: 1, 1: 1}

    if PARAMETERS["max_degree"] != 1:
        for feature in NUM_LIST:
            if feature in features_list_train:
                x_train_processed = polynomial_expansion(
                    x_train_processed,
                    feature,
                    features_list_train,
                    PARAMETERS["max_degree"],
                )
                x_test_processed = polynomial_expansion(
                    x_test_processed,
                    feature,
                    features_list_train,
                    PARAMETERS["max_degree"],
                )

    if PARAMETERS["cluster"]:
        centroids, labels = kmeans(x_train_processed, 2, 12)
        x_train_processed = np.concatenate(
            (x_train_processed, labels[:, np.newaxis]), axis=1
        )
        features_list_train.append("CLUSTER")
        distances_te = compute_distances(x_test_processed, centroids)
        labels_te = np.argmin(distances_te, axis=1)
        x_test_processed = np.concatenate(
            (x_test_processed, labels_te[:, np.newaxis]), axis=1
        )
        features_list_test.append("CLUSTER")

    if PARAMETERS["oversampling_ratio"] != 0:
        n_posi = (y_train_converted == 1).sum()
        n_nega = (y_train_converted == 0).sum()
        y_tr_posi_idx = np.argwhere(y_train_converted == 1).flatten()

        n_posi_to_sample = int(
            (PARAMETERS["oversampling_ratio"] * (n_posi + n_nega) - n_posi)
            / (1 - PARAMETERS["oversampling_ratio"])
        )
        posi_sample_idx = np.random.choice(
            y_tr_posi_idx, n_posi_to_sample, replace=True
        )

        x_tr_sample = x_train_processed[posi_sample_idx]
        y_tr_sample = y_train_converted[posi_sample_idx]

        x_tr = np.concatenate((x_train_processed, x_tr_sample), axis=0)
        y_tr = np.concatenate((y_train_converted, y_tr_sample), axis=0)

    initial_w = np.zeros(x_train_processed.shape[1], dtype=float)

    w, _ = elastic_net_logistic_regression_weighted(
        y_tr,
        x_tr,
        PARAMETERS["lambda"],
        initial_w,
        PARAMETERS["max_iter"],
        PARAMETERS["gamma"],
        class_weight,
        PARAMETERS["alpha"],
        PARAMETERS["batch_size"],
    )

    y_pred_te = np.where(np.dot(x_test_processed, w) >= PARAMETERS["threshold"], 1, -1)

    print(w.tolist())
    print(features_list_train)

    create_csv_submission(test_ids, y_pred_te, "final_final_submission.csv")
