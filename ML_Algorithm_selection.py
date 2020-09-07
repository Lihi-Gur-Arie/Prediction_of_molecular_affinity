#By Lihi Gur Arie

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn import tree
from termcolor import colored
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate

#########################################################################################################3

def run_over_models(train_data, cv=10, scaler = 'Standard', scale_by=(-100,100)):
    """
    Iterate over desired Machine Learning algorithms.
    The function will return the evaluation of all the examined models.

    Parameters
    ----------
    train_data  :{array-like, sparse matrix} of shape (n_samples, n_features+1)
        Training data with label at the last column.
    cv = int
        The number of k-fold for cross-validation
    scaler = str
        Scalar type. 'min_max' or 'Standard'
    scale_by = tuple (int,int)
        scale the data between (int, int)

    Returns
    -------
    results_table : pd.DataFrame
        A table containing the evaluations of all the models.
    """

    models_names = [ 'Ridge Regression','Lasso Regression', 'Gradient Boosting','Neural Network', 'XGBoost','AdaBoost', 'DecisionTree', 'BayesianRidge', 'SVR','Random Forest']
    models_list = [ Ridge (), Lasso(), GradientBoostingRegressor( random_state=1),MLPRegressor(learning_rate_init=0.005, max_iter=500, batch_size=64, random_state=1), xgb.XGBRegressor(objective="reg:squarederror", random_state=1, n_jobs=-1),AdaBoostRegressor( random_state=1),  tree.DecisionTreeRegressor(random_state=1, max_depth=6),  BayesianRidge(),  svm.SVR(),RandomForestRegressor(criterion = 'mse',random_state=1)]
    models_list = pd.Series(models_list, index=models_names)

    MSE_cv_results = pd.DataFrame(index=['MSE_full', 'MSE_threshold', 'MSE_variance', 'MAE_full', 'MAE_threshold','MAE_variance', 'r2'])

    results_table = pd.DataFrame()
    for i in range(len(models_list)):

        train_model = Model(train_data, threshold_percent=2, random_state=1)
        MSEs_train, MSEs_test = train_model.main(model_name=models_names[i], scaler = scaler,model=models_list.iloc[i],title=models_names[i], scale_by=scale_by, CV_Kfolds=cv)

        results_table = results_table.append(pd.Series(MSEs_test, name= models_names[i], index=['MSE','MSE_threshold','MSE_Variance', 'MAE','MAE_threshold', 'MAE_Variance','R2' ]))

    table = lambda MSE_cv_results: tabulate(results_table, headers='keys', tablefmt='psql')
    print(table(MSE_cv_results))

    return results_table

#####################################################################################################################################3
class Model:

    """
    This class scales, fit and evaluate a model.
    If final_test_data is not provided, the model's evaluation is the mean of cross validation results.
    """

    def __init__(self, data, threshold_percent, random_state, final_test_data = None):
        self.data = data.copy()
        self.final_test_data = final_test_data
        self.threshold_percent = threshold_percent
        self.random_state = random_state

    def main (self, model_name, model, title, scale_by, CV_Kfolds, scaler ):
        model_name = model_name
        self.model = model
        self.title = title
        self.scaler = scaler
        self.scale_by = scale_by
        self.CV_Kfolds = CV_Kfolds
        print(colored(model_name, 'green'))

        # Find the threshold value:
        self.find_threshold_value()

        if self.final_test_data is not None:
            MSEs_train, MSEs_test = self.fit_model(train_data = self.data, valid_data = self.final_test_data)

        else:
            MSEs_train, MSEs_test = self.cvKfold( cv=CV_Kfolds)

        return MSEs_train, MSEs_test


    def find_threshold_value (self):
        # Determine the threshold in the best % affinity scores
        threshold_index = int(self.data.shape[0] * (self.threshold_percent / 100))
        self.threshold_value = self.data.sort_values(by=['max_affin']).iloc[threshold_index, -1]

    def scale_data (self, train_data, valid_data):

        X_train = train_data.iloc[:, :-1]
        X_valid = valid_data.iloc[:, :-1]
        Y_train = np.array(train_data.iloc[:,-1]).reshape((-1, 1))
        Y_valid = np.array(valid_data.iloc[:,-1]).reshape((-1, 1))

        # Scale each i individually between the desired values
        if self.scaler == 'min_max':
            self.scalar_x = MinMaxScaler(feature_range=(self.scale_by[0], self.scale_by[1])).fit(X_train)
            self.scalar_y = MinMaxScaler(feature_range=(self.scale_by[0], self.scale_by[1])).fit(Y_train)

        elif self.scaler == 'Standard':
            self.scalar_x = StandardScaler().fit(X_train)
            self.scalar_y = StandardScaler().fit(Y_train)

        # Transform x and y:
        X_train = pd.DataFrame(self.scalar_x.transform(X_train), index=X_train.index, columns=X_train.columns)
        X_valid = pd.DataFrame(self.scalar_x.transform(X_valid), index=X_valid.index, columns=X_valid.columns)
        Y_train_scaled = self.scalar_y.transform(Y_train)
        Y_valid_scaled = self.scalar_y.transform(Y_valid)

        # Create a Data table with the features (X), the unscaled scores(y_true unscaled),
        # the scaled scores (y_true scaled) and the above/under threshold scores (y class)
        train_data = X_train.copy()
        train_data['Y_scaled'] = Y_train_scaled
        train_data['unscaled'] = Y_train
        train_data['class'] = np.array([1.0 if y > self.threshold_value else -1.0 for y in Y_train])
        valid_data = X_valid.copy()
        valid_data['Y_scaled'] = Y_valid_scaled
        valid_data['unscaled'] = Y_valid
        valid_data['class'] = np.array([1.0 if y > self.threshold_value else -1.0 for y in Y_valid])

        return train_data, valid_data

    def cvKfold (self, cv):

        cv_round = 1                                       # count the round of the cv
        results_train = pd.DataFrame()
        results_valid = pd.DataFrame()

        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        self.data['class'] = np.array([1.0 if y > self.threshold_value else -1.0 for y in self.data.iloc[:,-1]])

        for train_index, valid_index in splitter.split(self.data, self.data.iloc[:, -1]):

            train_results, valid_results = self.fit_model(train_data = self.data.iloc[train_index, :-1], valid_data = self.data.iloc[valid_index,:-1])
            results_train [f'CV {cv_round}'] = train_results
            results_valid [f'CV {cv_round}'] = valid_results
        return results_train.mean(axis=1), results_valid.mean(axis=1)


    def fit_model (self, train_data, valid_data):

        # Scale the data:
        if self.scale_by != False:
            train_data, valid_data = self.scale_data(train_data, valid_data)

        # Fit the model:
        self.model.fit(train_data.iloc[:, :-3], train_data.iloc[:, -3])

        # Evaluate:
        pred_train, results_train = self.evaluate(scaled_x_data = train_data.iloc[:,:-3], y_true = train_data.iloc[:,-2])
        pred_valid, results_valid = self.evaluate(scaled_x_data = valid_data.iloc[:,:-3], y_true = valid_data.iloc[:,-2])

        return results_train, results_valid

    def predict (self, data_to_predict):

        index_list = [x for x in data_to_predict.columns if x in self.final_test_data.columns]
        data_to_predict = data_to_predict[index_list]

        # Scale the new data
        scaled_x_data = self.scalar_x.transform(data_to_predict)

        # Predict with train model:
        scaled_pred_using_train_data = self.model.predict(scaled_x_data)
        scaled_pred_using_train_data = np.array(scaled_pred_using_train_data).reshape(-1, 1)
        pred_using_train_data = pd.DataFrame(self.scalar_y.inverse_transform(scaled_pred_using_train_data), index=[data_to_predict.index.values])

        return pred_using_train_data

    def evaluate (self, scaled_x_data, y_true):
        scaled_pred = self.model.predict(scaled_x_data)
        scaled_pred = np.array(scaled_pred).reshape(-1,1)
        pred = self.scalar_y.inverse_transform(scaled_pred)

        MSE_full = mean_squared_error(y_true, pred)
        MAE_full = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)

        # Calculate validation MSE to the best scores only:
        ys = pd.DataFrame({'y_true': y_true, 'y_pred': pred.squeeze()})                    # Create a new list with the best y_true scores (below threshold)
        ys = ys.sort_values(by=['y_true'])                                                 # sort scores by y_true
        ys = ys.loc[ys.y_true <= self.threshold_value]                                     # trim list by y_true threshold
        MSE_best_scores = mean_squared_error(ys.y_true, ys.y_pred)                         # calaculate MSE for best scores
        MAE_best_scores = mean_absolute_error (ys.y_true, ys.y_pred)                       # calaculate MAE for best scores

        results = pd.Series([MSE_full, MSE_best_scores, MAE_full, MAE_best_scores, r2], index=['MSE', 'MSE_threshold', 'MAE', 'MAE_threshold', 'R2'])

        return pred, results