"""Functions to preprocess the data for NaN handling, outliers handling, one-hot encoding."""

import numpy as np
import matplotlib.pyplot as plt


def select_features(
    x: np.array, features_list: list[str], header: list[str]
) -> np.array:
    """Returns the datasets with the chosen features only

    Args:
        x: shape=(N, D_total)
        features_list: list of features to keep of len D
        header: list of the features in the same order as the columns of x

    Returns:
        selected_x: shape=(N, D)
    """
    all_features_idx = [header.index(feature) for feature in features_list]
    selected_x = x[:, all_features_idx]
    return selected_x



def show_cross_corr(x: np.array, feature_list: list[str]) -> np.array:
    """Computes the cross-correlation of the features and plots the heatmap

    Args:
        x: shape=(N, D)
        feature_list: a list with the names of the features in the dataset

    Returns:
        cross_corr: the array with the cross correlation values
    """

    cross_corr = np.abs(np.corrcoef(x, rowvar = False))

    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(cross_corr)

    ax.set_xticks(np.arange(len(feature_list)), labels=feature_list)
    ax.set_yticks(np.arange(len(feature_list)), labels=feature_list)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")
    fig.colorbar(im)

    ax.set_title("cross_corr of features in dataset")
    fig.tight_layout()
    plt.show()

    return cross_corr



def preprocess_null_dataset(
    x: np.array,
    null_values_list: list[list[int]],
    columns_list: list[list[str]],
    features_list: list[str],
):
    """Replaces the specific values for missing answer of each feature with NaN

    Args:
        x: shape=(N, D)
        null_values_list: a list of the lists of specific values to replace
        columns_list: a list of the lists of features corresponding to the specific values to replace
    """
    for null_values, columns in zip(null_values_list, columns_list):
        columns_idx = [
            features_list.index(column) for column in columns if column in features_list
        ]
        x[:, columns_idx] = np.where(
            np.isin(x[:, columns_idx], null_values), np.nan, x[:, columns_idx]
        )




def process_diab(x: np.array, feature: str, features_list: list[str]):
    """Processes the DIABETE3 feature by only keeping Yes as the positive value

    Args:
        x: shape=(N, D)
        feature: the name of the feature DIABETE3
        features_list: the list of features in the dataset in the same order as the columns
    """
    x[:, features_list.index(feature)] = np.where(
        np.isin(x[:, features_list.index(feature)], [2, 3, 4]),
        0,
        x[:, features_list.index(feature)],
    )
    x[:, features_list.index(feature)] = np.where(
        np.isnan(x[:, features_list.index(feature)]),
        0,
        x[:, features_list.index(feature)],
    )


def process_rehab(x: np.array, feature: str, feature_list: list[str]):
    """Processes the STREHAB1 and HAREHAB1 features by keeping the Yes and No as the positive value since it means the patient had a heart attack or a stroke

    Args:
        x: shape=(N, D)
        feature: the name of the features STREHAB1 or HAREHAB1
        features_list: the list of features in the dataset in the same order as the columns
    """
    x[:, feature_list.index(feature)] = np.where(
        np.isin(x[:, feature_list.index(feature)], [1, 2]), 1, 0
    )


def process_caregiver(x: np.array, features: tuple[str], feature_list: list[str]):
    """Processes the CRGVREL1 and CRGVPRB1 features by returning 1 if the patient takes care of a blood relative with heart disease or hypertension

    Args:
        x: shape=(N, D)
        features: the tuple with the names of the features (CRGVREL1, CRGVPRB1)
        features_list: the list of features in the dataset in the same order as the columns
    """
    caregiver_related_vector = np.where(
        np.isin(x[:, feature_list.index(features[0])], [1, 2, 5, 9, 10, 11, 12, 13]),
        1,
        0,
    )
    caregiver_problem_vector = np.where(
        x[:, feature_list.index(features[1])] == 8, 1, 0
    )
    caregiver_vector = caregiver_related_vector * caregiver_problem_vector
    feature_list.append("CRGFAMH")
    x = np.concatenate((x, caregiver_vector[:, np.newaxis]), axis=1)
    x = np.delete(x, feature_list.index(features[0]), axis=1)
    feature_list.pop(feature_list.index(features[0]))
    x = np.delete(x, feature_list.index(features[1]), axis=1)
    feature_list.pop(feature_list.index(features[1]))


def one_hot_encoding(
    x: np.array, feature: str, features_list: list[str], drop_first: bool
) -> np.array:
    """Processes a categorical feature by applying one-hot encoding. It adds a binary feature for each category of the feature.

    Args:
        x: shape=(N, D)
        feature: the name of the feature to process
        features_list: the list of features in the dataset in the same order as the columns
        drop_first: a boolean denoting if the first category needs to be dropped to avoid an artificial correlation issue between features

    Returns:
        x: shape=(N, D + n_category - 2)
    """
    feature_vector = x[:, features_list.index(feature)]
    categories = np.unique(feature_vector[~np.isnan(feature_vector)])
    new_features = []
    for i, category in enumerate(categories):
        if i == 0 and drop_first:
            continue
        one_hot_vector = np.where(feature_vector == category, 1, 0)
        one_hot_vector = np.where(np.isnan(feature_vector), np.nan, feature_vector)
        x = np.concatenate((x, one_hot_vector[:, np.newaxis]), axis=1)
        features_list.append(feature + "_" + str(category))
        new_features.append(feature + "_" + str(category))
    if feature == "_CHOLCHK" and "_CHOLCHK_3.0" not in features_list:
        x = np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)
        features_list.append("_CHOLCHK_3.0")
    x = np.delete(x, features_list.index(feature), axis=1)
    features_list.pop(features_list.index(feature))
    return x


def binary_to_one_hot(
    x: np.array, binary_features_list: list[str], features_list: list[str]
):
    """Processes the binary features by setting the values to 1 and 0 in the same order as in the dataset

    Args:
        x: shape=(N, D)
        binary_features_list: the list of the names of the features to process
        features_list: the list of features in the dataset in the same order as the columns
    """
    feature_idx = [
        features_list.index(feature)
        for feature in binary_features_list
        if feature in features_list
    ]
    x[:, feature_idx] = x[:, feature_idx] - 1


def reverse_binary_to_one_hot(
    x: np.array, reverse_binary_features_list: list[str], features_list: list[str]
):
    """Processes the reverse binary features by setting the values to 1 and 0 in the opposite order as in the dataset

    Args:
        x: shape=(N, D)
        reverse_binary_features_list: the list of the names of the features to process
        features_list: the list of features in the dataset in the same order as the columns
    """
    feature_idx = [
        features_list.index(feature)
        for feature in reverse_binary_features_list
        if feature in features_list
    ]
    x[:, feature_idx] = np.where(x[:, feature_idx] == 2, 0, x[:, feature_idx])


def reverse_categorical(x: np.array, feature: str, features_list: list[str]):
    """Processes a reverse categorical feature by reversing the order of the categories provided in the dataset

    Args:
        x: shape=(N, D)
        feature: the name of the feature to process
        features_list: the list of features in the dataset in the same order as the columns
    """
    feature_vector = x[:, features_list.index(feature)]
    max_val = feature_vector[~np.isnan(feature_vector)].max()
    mapping = np.concatenate((np.array([0]), np.arange(max_val, 0, -1)))
    x[:, features_list.index(feature)] = np.where(
        np.isnan(feature_vector), np.nan, mapping[feature_vector.astype(int)]
    )




def tails_clipping(x: np.array, num_features: list[str], features_list: list[str]):
    """Handles outliers by clipping the numerical features with the 5th and 95th percentile

    Args:
        x: shape=(N, D)
        num_features: the list of the names of the numerical features to process
        features_list: the list of features in the dataset in the same order as the columns
    """
    num_idx = [
        features_list.index(feature)
        for feature in num_features
        if feature in features_list
    ]
    num_x = x[:, num_idx]
    fifth_percentile = np.percentile(num_x, 5, axis=0)
    ninety_fifth_percentile = np.percentile(num_x, 95, axis=0)
    x[:, num_idx] = np.clip(num_x, fifth_percentile, ninety_fifth_percentile)



def filling_nan_num(x: np.array, num_features: list[str], features_list: list[str]):
    """Handles the NaNs in the numerical features by filling in with the median values of the remaining datapoints

    Args:
        x: shape=(N, D)
        num_features: the list of the names of the numerical features to process
        features_list: the list of features in the dataset in the same order as the columns
    """
    num_idx = [
        features_list.index(feature)
        for feature in num_features
        if feature in features_list
    ]
    medians = np.nanpercentile(x[:, num_idx], 50, axis=0)
    for i, col in enumerate(num_idx):
        x[:, col] = np.nan_to_num(x[:, col], medians[i])


def filling_nan_cat(x: np.array, cat_features: list[str], features_list: list[str]):
    """Handles the NaNs in the categorical features by filling in with the most common category of the remaining datapoints

    Args:
        x: shape=(N, D)
        cat_features: the list of the names of the categorical features to process
        features_list: the list of features in the dataset in the same order as the columns
    """
    cat_idx = [
        features_list.index(feature)
        for feature in cat_features
        if feature in features_list
    ]
    non_nan_idx = ~np.isnan(x[:, cat_idx])
    for i, idx in enumerate(cat_idx):
        feature_vector = x[non_nan_idx[:, i], idx]
        most_common_cat = np.argmax(np.bincount(feature_vector.astype(int)))
        x[:, idx] = np.where(np.isnan(x[:, idx]), most_common_cat, x[:, idx])



def normalize_x(x: np.array):
    """Normalizes the dataset

    Args:
        x: shape=(N, D)
    """
    x_maxes = np.max(x, axis=0)
    x_mins = np.min(x, axis=0)
    x = (x - x_mins) / (x_maxes - x_mins)


def standardize_x(x: np.array, num_features_list: list[str], features_list: list[str]):
    """Standardizes the numerical features of the dataset

    Args:
        x: shape=(N, D)
        num_features_list: the list of the names of the numerical features to process
        features_list: the list of features in the dataset in the same order as the columns
    """
    num_feature_idx = [
        features_list.index(feature)
        for feature in num_features_list
        if feature in features_list
    ]
    x_means = np.mean(x[:, num_feature_idx], axis=0)
    x_stds = np.std(x[:, num_feature_idx], axis=0)
    x[:, num_feature_idx] = (x[:, num_feature_idx] - x_means) / x_stds




def preprocess_data(
    x: np.array,
    values_null_list: list[list[str]],
    columns_null_list: list[list[str]],
    features_list: list[str],
    num_list: list[str],
    binary_list: list[str],
    binary_reverse_list: list[str],
    categorical_label_list: list[str],
    reverse_categorical_label_list: list[str],
    one_hot_encoding_list: list[str],
) -> tuple[np.array, list[str]]:
    """Full preprocessing of the dataset with the previous functions

    Args:
        x: shape=(N, D)
        values_null_list: a list of the lists of specific values to replace
        columns_null_list: a list of the lists of features corresponding to the specific values to replace
        features_list: the list of features in the dataset in the same order as the columns
        num_list: the list of the names of the numerical features to process
        binary_list: the list of the names of the binary features to process
        binary_reverse_list: the list of the names of the reverse binary features to process
        categorical_label_list: the list of the names of the categorical features to process
        reverse_categorical_label_list: the list of the names of the reverse categorical features to process
        one_hot_encoding_list: the list of the names of the categorical features to process with one-hot encoding

    Returns:
        x: shape=(N, D_final)
        features_list: the list of features in the dataset in the same order as the columns after preprocessing
    """

    preprocess_null_dataset(x, values_null_list, columns_null_list, features_list)
    if "DIABETE3" in features_list:
        process_diab(x, "DIABETE3", features_list)
    if "HAREHAB1" in features_list:
        process_rehab(x, "HAREHAB1", features_list)
    if "STREHAB1" in features_list:
        process_rehab(x, "STREHAB1", features_list)
    if "CRGVPRB1" in features_list and "CRGVREL1" in features_list:
        process_caregiver(x, ["CRGVREL1", "CRGVPRB1"], features_list)

    if "_FLSHOT6" and "_PNEUMO2" in features_list:
        features_fill_nans = ["_FLSHOT6", "_PNEUMO2"]
        columns_fill_nans = [
            features_list.index(feature) for feature in features_fill_nans
        ]
        x[:, columns_fill_nans] = np.where(
            np.isnan(x[:, columns_fill_nans]), 0, x[:, columns_fill_nans]
        )
    elif "_FLSHOT6" in features_list:
        features_fill_nans = ["_FLSHOT6"]
        columns_fill_nans = [
            features_list.index(feature) for feature in features_fill_nans
        ]
        x[:, columns_fill_nans] = np.where(
            np.isnan(x[:, columns_fill_nans]), 0, x[:, columns_fill_nans]
        )
    elif "_PNEUMO2" in features_list:
        features_fill_nans = ["_PNEUMO2"]
        columns_fill_nans = [
            features_list.index(feature) for feature in features_fill_nans
        ]
        x[:, columns_fill_nans] = np.where(
            np.isnan(x[:, columns_fill_nans]), 0, x[:, columns_fill_nans]
        )

    filling_nan_num(x, num_list, features_list)
    filling_nan_cat(
        x,
        binary_list
        + binary_reverse_list
        + categorical_label_list
        + reverse_categorical_label_list
        + one_hot_encoding_list,
        features_list,
    )

    for feature in one_hot_encoding_list:
        if feature in features_list:
            x = one_hot_encoding(x, feature, features_list, True)

    tails_clipping(x, num_list, features_list)

    binary_to_one_hot(x, binary_list, features_list)
    reverse_binary_to_one_hot(x, binary_reverse_list, features_list)
    for feature in reverse_categorical_label_list:
        if feature in features_list:
            reverse_categorical(x, feature, features_list)

    num_list_idx = [
        features_list.index(feature)
        for feature in num_list
        if feature != "HTM4" and feature in features_list
    ]
    x[:, num_list_idx] = np.log(x[:, num_list_idx] + 0.001)
    standardize_x(x, num_list, features_list)

    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    features_list = ["BIAS"] + features_list

    return x, features_list
