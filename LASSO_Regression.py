# by Lihi Gur-Arie

import numpy as np
import pandas as pd
from sklearn.linear_model import  Lasso, LinearRegression
from termcolor import colored
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import pickle
import shap

class LassoRegressor:
    # This class preforms Lasso Regression.
    # Add Test data only if this is the final test.
    # The model will preform cross validation if there is no test data. If there is test data, it will train once on all the train data, and evaluate on the test data.
    # If there is Test data, the model will only run once on the whole training data,
    # and will be evaluated by the Test data.

    def __init__(self, data,  threshold_value=-11.24, final_test_data=None, random_state=1):

        self.data = data.copy()                                # Training data in Pandas DataFrame form
        self.final_test_data = final_test_data                 # Test data in Pandas DataFrame form
        self.random_state = random_state                       # The random state seed. Default = 1.
        self.threshold_value = threshold_value
        self.scalar_x = None
        self.scalar_y = None
        self.XYs = None
        self.title = None
        self.residuals = None
        self.scale_by = None
        self.scaler = None
        self.save_plots = None
        self.CV_Kfolds = None

    def main (self, title, scale_by, CV_Kfolds, scaler , alpha = 0.2, smote = False,  save_plots=None, show_plots = False):
        # Save variables:
        self.title = title
        self.scaler = scaler
        self.scale_by = scale_by
        self.CV_Kfolds = CV_Kfolds
        self.save_plots = save_plots
        self.show_plots = show_plots
        print(colored(self.title, 'green'))


        # If there is Test data, fit the model over all data in the training group.
        if self.final_test_data is not None:
            MSEs_train, MSEs_test, coefficients = self.fit_model(train_data = self.data, valid_data = self.final_test_data, smote = smote, cv_round = 1, alpha = alpha)

        # If no test data was introduced, perform 10 Kfold cross validation:
        else:
            MSEs_train, MSEs_test, coefficients = self.cvKfold( cv=CV_Kfolds, smote=smote, alpha = alpha)

        return MSEs_train, MSEs_test, coefficients

    def scale_data (self, train_data, valid_data):
    # Option are: 'min_max', 'Standard'

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

        print('Data is scaled')
        return train_data, valid_data

    def smote (self, train_data):
    # This function will add samples to the minority group, to compansate for imbalanced data

        # SMOTE transform the dataset
        X_smote_scaled, y_smote_scaled = SMOTE(random_state=self.random_state, n_jobs=-1).fit_resample(train_data.iloc[:,:-1], train_data.iloc[:,-1])

        train_data = X_smote_scaled.iloc[:, :-2].copy()
        train_data['Y_scaled'] = X_smote_scaled.iloc[:, -2]
        Y_train_scaled = np.array(X_smote_scaled.iloc[:, -2]).reshape(-1, 1)
        train_data['Y_unscaled'] = self.scalar_y.inverse_transform(Y_train_scaled)
        train_data['Y_class'] = (np.array([1.0 if y > self.threshold_value else -1.0 for y in train_data.Y_unscaled])).reshape(-1, 1)
        return train_data

    def cvKfold (self, cv, smote, alpha):
    # This function will split the data into tarin / validation in a stratified way
    # cv takes an integer representing the k fold.
    # The function returns the average results of all the k folds.

        cv_round = 1                                       # count the round of the cv

        results_train = pd.DataFrame()
        results_valid = pd.DataFrame()
        coefficients_list = pd.DataFrame()

        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        self.data['class'] = np.array([1.0 if y > self.threshold_value else -1.0 for y in self.data.iloc[:, -1]])

        for train_index, valid_index in splitter.split(self.data, self.data.iloc[:, -1]):
            print(colored(f'CV {cv_round}', 'cyan'))

            train_results, valid_results, coefficients = self.fit_model(train_data = self.data.iloc[train_index, :-1], valid_data = self.data.iloc[valid_index,:-1], smote = smote,  cv_round = cv_round, alpha = alpha)

            results_train [f'CV {cv_round}'] = train_results
            results_valid [f'CV {cv_round}'] = valid_results
            coefficients_list [f'CV {cv_round}'] = coefficients

            cv_round += 1

        coefficients_results = pd.DataFrame()
        coefficients_results['coef_mean'] = coefficients_list.mean(axis=1)
        coefficients_results['coef_median'] = coefficients_list.median(axis=1)
        coefficients_results['abs_coef_median'] = abs(coefficients_results['coef_median'])
        coefficients_results['coef_var'] = coefficients_list.var(axis=1)

        # Sort coefficiants by 'abs_coef_median' in descending order:
        coefficients_results.sort_values(by=['abs_coef_median'], inplace=True, ascending=False)

        return results_train.mean(axis=1), results_valid.mean(axis=1), coefficients_results

    def fit_model (self, train_data, valid_data, smote,  cv_round, alpha):
        if alpha != 0:
            self.model = Lasso (alpha = alpha, max_iter = 50000)
        else:
            self.model = LinearRegression()

        if self.scale_by != False:
            train_data, valid_data = self.scale_data(train_data, valid_data)

        print(f'SMOTE = {smote}')
        if smote == True:
            train_data = self.smote(train_data)

        # Fit the model:
        history = self.model.fit(train_data.iloc[:, :-3], train_data.iloc[:, -3])
        print ('Model is fitted')

        pred_train, results_train = self.evaluate(scaled_x_data = train_data.iloc[:,:-3], y_true = train_data.iloc[:,-2])
        pred_valid, results_valid = self.evaluate(scaled_x_data = valid_data.iloc[:,:-3], y_true = valid_data.iloc[:,-2])

        # The coefficients for each i:
        coefficients = pd.Series(self.model.coef_, index=train_data.iloc[:,: -3].columns)

        if self.show_plots == True:
            # residuals plot:
            self.plot_residuals (y_true_train = train_data.iloc[:, -2],train_pred = pred_train,R_squered_train=results_train['R2'], mse_train=results_train['MSE'],
                         y_true_valid= valid_data.iloc[:, -2], valid_pred = pred_valid, cv=cv_round,
                         mse_valid=results_valid['MSE'], R_squered_valid =results_valid['R2'])

            # expected vs. observed plot:
            self.plot_ys(y_true_train=train_data.iloc[:, -2], train_pred = pred_train, R_squered_train = results_train['R2'], y_true_valid=valid_data.iloc[:, -2],
                         valid_pred=pred_valid, R_squered_valid =results_valid['R2'])

            # shap_force_plot plots:
            self.shap_plot(X_train = train_data.iloc[:, :-3], Y_train=train_data.iloc[:, -3])

        return results_train, results_valid, coefficients


    def features_report(self, X_train, Y_train, global_shap_values):
        data = X_train.copy()
        data['max_affin'] = Y_train
        correlation_map = data.corr()

        self.features_analysis = pd.DataFrame(index=self.data.columns)
        self.features_analysis['Features'] = self.data.columns.values
        self.features_analysis['Correlation_to_Y'] = correlation_map.iloc[-1]
        self.features_analysis['Correlation_to_Y_abs'] = abs(self.features_analysis['Correlation_to_Y'])

        self.features_analysis = self.features_analysis.iloc[:-1]
        self.features_analysis['Global_Shap_Value'] = global_shap_values
        self.features_analysis['Coefficients'] = self.model.coef_
        self.features_analysis['Coefficients_abs'] = abs(self.features_analysis['Coefficients'] )

        # Save to exel file:
        with pd.ExcelWriter(f'Features_Analysis_Report_{self.title}.xlsx') as writer:
            self.features_analysis.to_excel(writer, sheet_name='features_analysis')
            correlation_map.to_excel(writer, sheet_name='Correlation_map')

        ##### PLOTS  #####

        # Shap values Plot:
        self.features_analysis = self.features_analysis.sort_values(by=['Coefficients_abs'], ascending=False)  # sort by shap values
        sns.set_style("darkgrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 27))
        ax = sns.barplot(x='Global_Shap_Value', y='Features', data=self.features_analysis, ax=axes[0])
        ax.set_xlabel('Global_Shap_Value', fontsize=30)
        ax.set_ylabel("Features", fontsize=30)
        ax.set_yticklabels(labels=self.features_analysis.Features, fontsize=20)

        # Coefficients plot:
        ax = sns.barplot(x='Coefficients', y='Features', data=self.features_analysis, ax=axes[1])
        ax.set_xlabel('Regression coefficients', fontsize=30)

        # Correlation_to_Y Plot:
        ax = sns.barplot(x='Correlation_to_Y', y='Features', data=self.features_analysis, ax=axes[2])
        ax.set_xlabel('Correlation_to_Y', fontsize=30)

        # Save plots:
        if self.save_plots == True:
            plt.savefig(f"Features_Analysis_plot_{self.title}.png", bbox_inches='tight', dpi=600)
            print ('Features_Analysis_plot was saved')
        plt.show()
        w=1

    def shap_plot(self, X_train, Y_train):

        explainer = shap.LinearExplainer(self.model, X_train)   # The data here suppose to be the data that the model was train on
        shap_values = explainer.shap_values(X_train)
        global_shap_values = np.abs(shap_values).mean(0)

        # Get Features report:
        self.features_report(X_train, Y_train, global_shap_values)

        # Summary plot:
        plt.figure()
        shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False, max_display = 45)
        plt.savefig(f"Summary_plot.png", bbox_inches='tight', dpi=600)
        plt.show()

        # Bar summary plot:
        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display = 45)
        plt.savefig(f"bar_plot.png", bbox_inches='tight', dpi=600)
        plt.show()


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

        # Un-scale predictions:
        scaled_pred = self.model.predict(scaled_x_data)
        scaled_pred = np.array(scaled_pred).reshape(-1,1)
        pred = self.scalar_y.inverse_transform(scaled_pred)

        # Evaluation:
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

    def plot_residuals(self, y_true_train,train_pred,R_squered_train,mse_train,y_true_valid, valid_pred, cv, mse_valid,R_squered_valid):

        # Residuals Plot:
        sns.set(color_codes=True)
        residuals_train = y_true_train - train_pred.squeeze()
        residuals_valid = y_true_valid - valid_pred.squeeze()
        plt.plot(train_pred.squeeze(), np.zeros(len(train_pred)), color='black')
        sns.scatterplot(x=train_pred.squeeze(), y=residuals_train.squeeze(),color='lightgreen' )
        sns.scatterplot(x=valid_pred.squeeze(), y=residuals_valid.squeeze(), legend="brief")
        plt.title(f"Residuals plot", fontsize=18)
        plt.xlabel("Predicted binding Score", fontsize=14)
        plt.ylabel("Residuals (Observed - Predicted)", fontsize=14)
        plt.legend([f'Perfect prediction R^2 = 1',f'Train data R^2 = {round(R_squered_train,2)}, MSE = {round(mse_train, 3)}',f'Test  data  R^2 = {round(R_squered_valid,2)},  MSE = {round(mse_valid, 3)}'],
            loc='lower left')

        # Save plot:
        if self.save_plots == True:
            plt.savefig(f'N_Residuals_plot_{self.title}_cv_{cv}.png')
        plt.show()
        print('Plot is ready')
        return self.residuals

    def plot_ys(self, y_true_train, train_pred, R_squered_train, y_true_valid, valid_pred, R_squered_valid):

        # Plot predictions vs. Y_true
        sns.set(color_codes=True)
        sns.regplot(x=train_pred, y=y_true_train, ci=0, color='lightgreen' ,truncate=False, scatter_kws={'label': f'Train molecules'},
                    line_kws={'lw': 3,'label': f'Train data: R^2 = {round(R_squered_train, 3)}'}, n_boot=1000)
        sns.regplot(x=valid_pred, y=y_true_valid, ci=0,  truncate=False, scatter_kws= {'label': f'Test molecules'},line_kws = {'label': f'Test  data: R^2 = {round(R_squered_valid,3)}'}, n_boot = 1000)
        plt.title(f"Predictions Evaluation on Test data", fontsize=18)
        plt.xlabel("Predicted binding Score", fontsize=14)
        plt.ylabel("Observed bindind Score", fontsize=14)
        plt.legend(loc='upper left')

        if self.save_plots == True:
            plt.savefig(f'N_Ys_plot_{self.title}.png')
        plt.show()
        print ('Plot is ready')

    def save_model(self, file_name):

        # save model:
        filename = f'Saved_model_{file_name}.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        print ('Model have been saved')
        # save scalar:
        pickle.dump(self.scalar_x, open(f'Saved_scaler_x_{file_name}.sav', 'wb'))
        pickle.dump(self.scalar_y, open(f'Saved_scaler_y_{file_name}.sav', 'wb'))
        print('Scaler have been saved')
        # Save features names:
        pd.Series(self.final_test_data.columns).to_csv(r'C:\Users\froma\Desktop\ML\Project Korona\Lihi\Saved_features_names_model_5.csv')
        print ('Features have been saved')
