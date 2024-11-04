[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MqChnODK)

MICHDA Classification Project
====================
This repository contains the report and code for the first project of Machine Learning course (CS-433).


## Team Components
For any question and/or curiosity, feel free to reach
* [Valentin Aolaritei](mailto:valentin.aolaritei@epfl.ch)
* [Alberto De Laurentis](mailto:alberto.delaurentis@epfl.ch)
* [Martin Louis Le Bras](mailto:martin.lebras@epfl.ch)

## Dataset

The datasets ```x_train.csv```, ```y_train.csv``` and ```x_test.csv``` can be found in the [ML course repository]([https://github.com/epfml/ML_course](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset.zip)) and should be loaded in this repository in a folder named "dataset".

## Code structure
1. [helpers.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/helpers.py) contains functions for downloading the data and creating the cv for submission;
2. [implementations.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/implementations.py) contains the required functions which we had to implement;
3. [preprocessing.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/preprocessing.py) functions to preprocess the data for NaN handling, outliers handling;
4. [feature_eng.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/feature_eng.py) contains functions for feature engineering;
5. [metrics.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/metrics.py) contains the functions used to compute the metrics;
6. [full_features_nan_and_cross_correlation_analysis.ipynb](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/full_features_nan_and_cross_correlation_analysis.ipynb) contains the code used for data analysis on all the dataset;
7. [nan_and_cross_correlation_analysis.ipynb](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/nan_and_cross_correlation_analysis.ipynb) contains the code used for data analysis only on some relevant features of the dataset;
8. [best_models.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/best_models.py) contains functions to find the best model with a grid search looping on several hyperparameters and reporting the k-fold cross validation F1-Score for each combination. It gives a JSON file named [results_per_parameters_sgd.json](https://github.com/CS-433/ml-project-1-mva-ml/blob/ccfd696448a875033d4f5d179afff8243a3a7533/results_per_parameters_sgd.json), which contains the best parameters we had found. The optimal parameter configuration is defined in [run.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/run.py), under the variable name ```PARAMETERS```.
9. [run.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/run.py) is the final code which will return the best submission in a CSV filled named ```final_final_submission.csv``` and absolute weights of final features. In case it does not work, [run.ipynb](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/run.ipynb) can be used. A copy of the best result achieved on AIcrowd (the one displayed by running [run.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/run.py) or [run.ipynb](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/run.ipynb)) is contained in the CSV filled named [Best_Model_Submission_Test.csv](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/Best_Model_Submission_Test.csv).


## Usage

You need to clone this repository.

Before running the code you will need to download the dataset ```x_train.csv```, ```y_train.csv``` and ```x_test.csv``` from [ML course repository]([https://github.com/epfml/ML_course](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset.zip)) and store them in a folder called ```dataset```.

Finally you can run either [run.py](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/run.py) or [run.ipynb](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/run.ipynb) to get submissions.

## Results

Our best submissions used the logistic regression and obtained a F1 score of 0.405 with an accuracy of 0.859 on AIcrowd test set.

Our submission number is #273445 on AIcrowd.


## Details

For details about the methods and the processes implemented in this project, [here](https://github.com/CS-433/ml-project-1-mva-ml/blob/main/report.pdf) you can find our report.