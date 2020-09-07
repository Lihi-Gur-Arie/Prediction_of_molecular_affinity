# By Lihi Gur-Arie


from Lihi.Lasso_Regressor import  LGRF,  call_LGRS
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import numpy as np
import pandas as pd
from sklearn.linear_model import  Lasso, Ridge
from termcolor import colored
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import inspect
import statsmodels.formula.api as sm
from sklearn.feature_selection import f_regression, mutual_info_regression
import seaborn as sns
from openpyxl import Workbook
import pickle
from tqdm import tqdm
import shap

########## Feature selection by correlation ##############################################################

class Remove_features_by_correlation:
    """
    Remove features by:
    1. Low Variance
    2. Low Correlation to Y
    3. Co-linearity to other features.

    Parameters
    ----------
    data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Training data, and label at the last column.
    """

    def __init__(self, data):
        self.data = data.copy()
        self.var_threshold = None
        self.y_cor_threshold = None
        self.x_cor_threshold = None
        print(f'The initial number of features is {data.shape[1] - 1}')

    def main(self, var_threshold, y_cor_threshold, x_cor_threshold):
        """
        Parameters
        ----------
        var_threshold  : float
            The variance threshold. Features that have lower variance then the threshold will be removed.
        y_cor_threshold : float
            The threshold of the correlation to Y. Features below the threshold will be removed.
        x_cor_threshold : float
            The threshold of the correlation to other features.
            Pairs of features above the threshold will be examined for their correlation to Y.
            The feature that has lower corelassion to Y between the two, will be removed.

        Returns
        -------
        train_data : pd.DataFrame
            The new train_data with reduced features
        """

        self.var_threshold = var_threshold
        self.y_cor_threshold = y_cor_threshold
        self.x_cor_threshold = x_cor_threshold

        # Remove features with low variance:
        selector = VarianceThreshold(var_threshold)
        selector.fit(self.data)
        train_data = self.data.loc[:, selector.get_support()]
        print(f'Low variance features were removed. New number of features = {train_data.shape[1] - 1}')

        # Remove features with low correlation to y:
        correlation_map_abs = abs(train_data.corr())                                          # Create a correlation map Using Pearson Correlation
        correlation_to_Y = correlation_map_abs.iloc[:, -1]                                    # Correlation with y_true
        train_data = train_data[correlation_to_Y.index[correlation_to_Y >= y_cor_threshold]]  # Selecting highly correlated features to y_true in the train data
        print(f'Features that are not correlated to y_true were reduced. New feature Number = {train_data.shape[1] - 1}')

        # Remove features with high correlation to each other (features that are highly dependent)
        correlation_map_abs = abs(train_data.corr())                                          # update correlation map after features reduction
        correlation_to_Y = correlation_map_abs.iloc[:, -1]                                    # update correlation to y_true after features reduction

        index_to_remove = []
        for i in correlation_map_abs.index:                                                   # for each i:
            correlation_list = correlation_map_abs.loc[i]                                     # Create a correlation list
            correlation_list.loc[i] = 0                                                       # change the correlation of a i to itself to 0 instead of 1
            correlation_list = correlation_list.iloc[:-1]                                     # remove y

            # Selecting highly correlated features in the train data
            collinear_index_list = correlation_list.index[correlation_list >= x_cor_threshold]  # Get a list of index above or equal to correlation threshold

            if len(collinear_index_list) > 0:                                                 # if there are any features above or equal to threshold:

                # Remove one of the two collinear features. Remove the one with the lower correlation to y
                for j in collinear_index_list:                                                # for each of the collinear features, leave the one with the better y correlation
                    if correlation_to_Y.loc[j] > correlation_to_Y.loc[i]:
                        index_to_remove.append(i)
                    else:
                        index_to_remove.append(j)

        index_to_remove = list(set([x for x in index_to_remove if x in train_data.columns]))

        train_data = train_data.drop(index_to_remove, axis=1)

        print(f'Features that are highly correlated to each other were reduced. New feature Number = {train_data.shape[1]}')

        return train_data

####################################################################

class Remove_features_by_Forward_feature_addition:
    """
    Remove features by Forward_feature_addition step.
        The best features are determined by lowest MSE results
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
            Training data with label at the last column.
        """
        self.data = data.copy()

    def main(self):
        """
        Returns
        -------
        train_data : pd.DataFrame
            The new train_data with reduced features
        """

        # Select the best features:
        full_results_table, selected_features_table = self.reduce_features()

        # Save the relevant features:
        best_features = pd.Series(selected_features_table.index[:np.argmin(selected_features_table['MSE']) + 1])
        best_features[len(best_features)] = 'max_affin'                                                      # Add 'max_affin' (y_true) to the features list

        print(f'Number of features before reduction = {self.data.shape[1]-1}')
        # Remove irrelevant features:
        train_data = self.data[best_features]
        print(f'Number of features after reduction = {train_data.shape[1]-1}')

        return train_data, full_results_table, selected_features_table, best_features

    def reduce_features(self):
        full_results_table = pd.DataFrame()
        selected_features_table = pd.DataFrame()

        counter = 1
        features_left_names = self.data.columns[:-1]

        for i in tqdm(range (self.data.shape[1] - 1)):

            all_results, new_feature = self.scan_features(selected_features_names = selected_features_table.index.array, counter=counter, features_left_names = features_left_names)
            all_results['Round'] = counter
            # Append the i with the best mse score:
            selected_features_table = selected_features_table.append(new_feature)
            full_results_table = full_results_table.append(all_results)
            # Remove features that are already chosen:
            features_left_names = features_left_names.drop(selected_features_table.index[-1])                # Append the i with the best Adj_r2 score
            counter += 1
        return full_results_table, selected_features_table

    def scan_features(self, selected_features_names, counter, features_left_names):
        results_table_train = pd.DataFrame()
        results_table_valid = pd.DataFrame()
        i=0
        for j in features_left_names:
            if i%20==0:
                print('i-',i,'c-' ,counter,'j-', j)
            i+=1
            if counter == 1:
                current_data = self.data[[ j, 'max_affin']]
            else:
                idx = np.concatenate((selected_features_names, j, 'max_affin'), axis=None)
                current_data = self.data[idx]

            model = LGRF(current_data, threshold_percent=10, random_state=1)
            results_train, results_valid, coefficiants = model.main(title=f'model16', scale_by=(-100, 100), CV_Kfolds=10, alpha=0.6, scaler='min_max', sample_weights=False, smote=False,  pca_components=False, keep_best_features=False,save_plots=False)
            results_valid.name = j
            results_table_train = results_table_train.append(pd.DataFrame(results_train).T)
            results_table_valid = results_table_valid.append(pd.DataFrame(results_valid).T)

        best_feature_index = np.argmin(results_table_valid['MSE'])                                           # The index of the i with the best mse
        best_feature_results = (results_table_valid.iloc[best_feature_index])                                # Get the results of the best i
        best_feature_results = best_feature_results.rename (results_table_valid.index[best_feature_index])   # Give Name to the i
        print (f'Best feature is {best_feature_results.name}')
        return results_table_valid, best_feature_results

###############################################################################################3

def reduce_features_by_lasso_coefficient (train_data):
    """
    This function reduce features by LASSO coefficient.
    It will run LASSO on the data, extract LASSO coefficients, remove the feature with the lowest coefficient
    and will run LASSO again iteratively. The best Features combination as determined by MSE, will be returned
    as the new train data.


    Parameters
    ----------
    train_data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Training data with label at the last column.

    Returns
    -------
    train_data : pd.DataFrame
        The new train_data with reduced features

    best_features: pd.Series
        The new features names, after the feature reduction

    results_table_valid: pd.DataFrame
        The evaluation results of the validation data

    results_table_train: pd.DataFrame
        The evaluation results of the training data

    coefficiant_abs_median: pd.DataFrame
        The median cross validation coefficients
    """
    print(f'Number of features is {train_data.shape[1] - 1}')

    results_table_train = pd.DataFrame()
    results_table_valid = pd.DataFrame()
    coefficiant_abs_median = pd.DataFrame()

    feature_names = train_data.columns.values
    last_reduced_feature = 'All_features'

    for i in range (train_data.shape[1] - 1):

        print(f'{len(feature_names)-1} Features')

        data_origin = train_data.copy().loc[:, feature_names]

        # Get the coefficients with default parameters:
        MSEs_train, MSEs_valid, coefficients = call_LGRS(data=data_origin, alpha=0.6, smote=False)
        results_table_train = results_table_train.append(pd.Series(MSEs_train, name=last_reduced_feature))
        results_table_valid = results_table_valid.append(pd.Series(MSEs_valid, name=last_reduced_feature))
        coefficiant_abs_median = coefficiant_abs_median.append(pd.Series(coefficients['abs_coef_median'], name=last_reduced_feature))

        # Reduce the feature with the lowest coeff:
        feature_names = coefficients.index.values[:-1]                                                    # Remove the worst feature
        last_reduced_feature = coefficients.index.values[-1]                                              # Update the name of the feature reduced
        feature_names = np.append(feature_names, 'max_affin')                                             # Add the y_true back

    feature_names = train_data.columns.values
    bad_features = results_table_valid.index[:np.argmin(results_table_valid['MSE'])+1]
    best_features = pd.Series([x for x in feature_names if x not in bad_features])

    # Trim the unselected features:
    train_data_new = train_data[best_features.squeeze()]

    return train_data_new, best_features, results_table_valid, results_table_train, coefficiant_abs_median


####################################################################################

def remove_features_by_name(train_data, test_data, features, alpha=1e-05,scaler='Standard'):

    """
    This function reduce features by feature's name, and return the new MSE score

    Parameters
    ----------
    train_data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Training data with label at the last column.

    test_data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Test data with label at the last column.

    features : list
        The features to be examined and removed

    alpha : float
        The penalty parameter for LASSO

    scaler : str
        The scaler type to use

    Returns
    -------
    train_data : pd.DataFrame
        The new train_data with reduced features

    test_data : pd.DataFrame
        The new test_data with reduced features

    mse_results: pd.DataFrame
        The evaluation results of after features reduction
    """
    if len(features)> 0:
        train_data = train_data.copy().drop(features, axis=1)
        test_data = test_data.copy().drop(features, axis=1)

    MSEs_train, MSEs_valid, coefficients = call_LGRS(data=train_data, alpha=alpha, scaler=scaler, smote=False, save_plots=False, show_plots=False)
    MSEs_train_test, MSEs_test, coefficients_test = call_LGRS(data=train_data, final_test_data=test_data, alpha=1e-05, scaler='Standard', smote=False, save_plots=False, show_plots=False)
    mse_results = pd.Series()
    mse_results['MSE_train'] = MSEs_valid['MSE']
    mse_results['MSE_test'] = MSEs_test['MSE']

    return train_data, test_data, mse_results

#######################################################################################

def backward_features_reduction(data):

    """
    Remove features by Backward_feature_removal step.
        The best features combination are determined by lowest MSE results

    Parameters
    ----------
    train_data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Training data with label at the last column.

    Returns
    -------
    final_data : pd.DataFrame
        The new train_data with reduced features

    results_valid : pd.DataFrame
        The evaluation results of after features reduction
    """
    results_valid = pd.DataFrame()

    MSEs_train, MSEs_valid, coefficients = call_LGRS(data=data, alpha=0.2, scaler='min_max',scale_by=(-200, 200))
    results_valid = results_valid.append(pd.Series(MSEs_valid, name='All_features'))
    ##################################################
    def remove_each_feature (data):

        backward_results_valid = pd.DataFrame()

        for feature in range(data.shape[1]-1):
            current_data = data.copy().drop(data.columns.values[feature], axis=1)
            features_name = data.columns.values[feature]

            MSEs_train, MSEs_valid, coefficients = call_LGRS(data=current_data, alpha=0.2, scaler ='min_max', scale_by = (-200,200))
            backward_results_valid = backward_results_valid.append(pd.Series(MSEs_valid, name=features_name))

        feature_to_remove = backward_results_valid.MSE.idxmin()
        new_data = data.copy().drop(feature_to_remove, axis=1)
        return new_data, backward_results_valid.loc[feature_to_remove]

    curr_data = data.copy()
    for j in  tqdm (range (data.shape[1]-2)):
        print (j)
        new_data, backward_results_valid = remove_each_feature (curr_data)
        results_valid = results_valid.append(backward_results_valid)
        curr_data = new_data

    min_mse = results_valid.MSE.idxmin()
    remove_list = (results_valid.loc[:min_mse]).index.values
    final_data = data.copy().drop(remove_list[1:], axis=1)

    return final_data, results_valid

##########################################################################################

def RFE_feature_seletion(train_data, valid_data, best_n_features):
    """
    Remove features by RFE.
    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features.

    Parameters
    ----------
    train_data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Training data with label at the last column.
    valid_data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Validation data with label at the last column..
    best_n_features: int
        Amount of features to keep

    Returns
    -------
    train_data : pd.DataFrame
        The new train_data with reduced features
    valid_data : pd.DataFrame
        The new valid_data with reduced features
    selected_features : list
    feature_ranking : list
    """

    model = Lasso (alpha = 0.6)
    rfe = RFE(model, best_n_features)
    fit = rfe.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])  # train the model

    Features_names = list(train_data.columns)
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), Features_names)))

    selected_features = fit.support_
    feature_ranking = fit.ranking_

    best_features_index = np.where(selected_features)[0]
    X_train_best_features = train_data.iloc[:,best_features_index]
    X_valid_best_features = valid_data.iloc[:, best_features_index]
    train_data = np.concatenate((X_train_best_features, train_data.iloc[:,-1:]), axis=1)
    valid_data = np.concatenate((X_valid_best_features, valid_data.iloc[:,-1:]), axis=1)
    return pd.DataFrame(train_data), pd.DataFrame(valid_data), selected_features,  feature_ranking