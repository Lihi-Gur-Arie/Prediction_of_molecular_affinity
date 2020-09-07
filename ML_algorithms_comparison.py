#By Lihi Gur Arie

import numpy as np
import pandas as pd
from sklearn.linear_model import  Lasso, Ridge
from termcolor import colored
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import inspect
from openpyxl import Workbook
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn import tree
from termcolor import colored
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tabulate import tabulate
import statistics
import seaborn as sns
import shap

#########################################################################################################3

def run_over_models(train_data, cv=10, scaler = 'Standard', scale_by=(-100,100), smote = False, sample_weights=False ):
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
    smote = bool
        If True, smote algorithm will be used to genarete new data of the minority class
    sample_weights = bool
        If True, Sample weights will be used.
        Warning - not all predictors can use sample weights

    Returns
    -------
    results_table : pd.DataFrame
        A table containing the evaluations of all the models.
    """

    RF = RandomForestRegressor(max_depth=10, max_features=24, random_state=1)

    models_names = [ 'Ridge Regression','Model5_Lasso Regression', 'Gradient Boosting ls', 'Gradient Boosting lad','Neural Network', 'XGBoost','AdaBoostRegressor_linear','AdaBoostRegressor_sequred',  'DecisionTree', 'Adaboost-RF',  'BayesianRidge',  'SVR','RF mse']
    models_list = [ Ridge ( alpha = 0.2, max_iter=40000, random_state=1), Lasso(alpha = 1e-5, max_iter=20000), GradientBoostingRegressor( loss = 'ls',learning_rate=0.3, max_depth=3, n_estimators=100, random_state=1), GradientBoostingRegressor( loss = 'lad',learning_rate=0.3, max_depth=3, n_estimators=100, random_state=1),MLPRegressor(learning_rate_init=0.005, max_iter=500, batch_size=64, random_state=1), xgb.XGBRegressor(objective="reg:squarederror", learning_rate = 0.2,  max_depth = 4, n_estimators = 100, random_state=1, n_jobs=-1),AdaBoostRegressor(learning_rate=0.01, loss ='linear', random_state=1),AdaBoostRegressor(learning_rate=0.01, loss ='square', random_state=1),  tree.DecisionTreeRegressor(random_state=1, max_depth=6), AdaBoostRegressor(base_estimator=RF, learning_rate=1, loss='linear', n_estimators=50,random_state=2),  BayesianRidge(),  svm.SVR(),RandomForestRegressor(criterion = 'mse',random_state=1)]
    models_list = pd.Series(models_list, index=models_names)

    MSE_cv_results = pd.DataFrame(index=['MSE_full', 'MSE_threshold', 'MSE_variance', 'MAE_full', 'MAE_threshold','MAE_variance', 'r2'])

    results_table = pd.DataFrame()
    for i in range(len(models_list)):

        train_model = Model(train_data, threshold_percent=2, random_state=1)
        MSEs_train, MSEs_test = train_model.main(model_name=models_names[i], scaler = scaler,model=models_list.iloc[i],title=models_names[i], scale_by=scale_by, CV_Kfolds=cv,
                                   sample_weights=sample_weights, smote=smote,  pca_components=False, keep_best_features=False)

        results_table = results_table.append(pd.Series(MSEs_test, name= models_names[i], index=['MSE','MSE_threshold','MSE_Variance', 'MAE','MAE_threshold', 'MAE_Variance','R2' ]))

    table = lambda MSE_cv_results: tabulate(results_table, headers='keys', tablefmt='psql')
    print(table(MSE_cv_results))

    return results_table

#####################################################################################################################################3
class Model:

    def __init__(self, data,  threshold_percent, random_state, final_test_data = None):
        self.data = data.copy()
        self.final_test_data = final_test_data
        self.threshold_percent = threshold_percent
        self.threshold_value = None
        self.random_state = random_state
        self.scalar_x = None
        self.scalar_y = None
        self.XYs = None
        self.title = None
        self.residuals = None
        self.scale_by = None
        self.scaler = None
        self.save_plots = None
        self.CV_Kfolds = None

    def main (self, model_name, model, title, scale_by, CV_Kfolds, scaler , alpha = 0.2,sample_weights = False, smote = False, pca_components = False, keep_best_features = False, save_plots=None, show_plots = False):
        model_name = model_name
        self.model = model
        self.title = title
        self.scaler = scaler
        self.scale_by = scale_by
        self.CV_Kfolds = CV_Kfolds
        self.save_plots = save_plots
        self.show_plots = show_plots
        print(colored(model_name, 'green'))

        # Find the threshold value:
        self.find_threshold_value()

        if self.final_test_data is not None:
            MSEs_train, MSEs_test = self.fit_model(train_data = self.data, valid_data = self.final_test_data,  sample_weights= False, smote = smote, cv_round = 1, alpha = alpha, pca_components = pca_components,keep_best_features = keep_best_features)

        else:
            MSEs_train, MSEs_test = self.cvKfold( cv=CV_Kfolds, sample_weights=sample_weights, smote=smote, alpha = alpha,  pca_components = pca_components, keep_best_features = keep_best_features)

        return MSEs_train, MSEs_test


    def find_threshold_value (self):
        # Determine the threshold in the 10% best affinity scores
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

    def smote (self, train_data):

        # transform the dataset
        X_smote_scaled, y_smote_scaled = SMOTE(random_state=self.random_state, n_jobs=-1).fit_resample(train_data.iloc[:,:-1], train_data.iloc[:,-1])
        train_data = X_smote_scaled.iloc[:, :-2].copy()
        train_data['Y_scaled'] = X_smote_scaled.iloc[:, -2]
        Y_train_scaled = np.array(X_smote_scaled.iloc[:, -2]).reshape(-1, 1)
        train_data['Y_unscaled'] = self.scalar_y.inverse_transform(Y_train_scaled)
        train_data['Y_class'] = (np.array([1.0 if y > self.threshold_value else -1.0 for y in train_data.Y_unscaled])).reshape(-1, 1)
        return train_data

    def cvKfold (self, cv, sample_weights, smote, alpha, pca_components, keep_best_features):

        cv_round = 1                                       # count the round of the cv

        results_train = pd.DataFrame()
        results_valid = pd.DataFrame()

        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        self.data['class'] = np.array([1.0 if y > self.threshold_value else -1.0 for y in self.data.iloc[:,-1]])

        for train_index, valid_index in splitter.split(self.data, self.data.iloc[:, -1]):

            train_results, valid_results = self.fit_model(train_data = self.data.iloc[train_index, :-1], valid_data = self.data.iloc[valid_index,:-1],   sample_weights = sample_weights, smote = smote,  cv_round = cv_round, alpha = alpha,  pca_components = pca_components, keep_best_features = keep_best_features)
            results_train [f'CV {cv_round}'] = train_results
            results_valid [f'CV {cv_round}'] = valid_results

        return results_train.mean(axis=1), results_valid.mean(axis=1)


    def fit_model (self, train_data, valid_data, sample_weights, smote,  cv_round, alpha, pca_components = False,  keep_best_features = False):

        if self.scale_by != False:
            train_data, valid_data = self.scale_data(train_data, valid_data)

        if keep_best_features != False:
            train_data, valid_data, selected_features, feature_ranking = self.feature_importance_rfe(train_data = train_data, valid_data = valid_data, best_n_features = keep_best_features)

        if pca_components != False:
            train_data, valid_data  = self.pca(n_components = pca_components, train_data = train_data, valid_data = valid_data)

        if smote == True:
            train_data = self.smote(train_data)

        if sample_weights == True:
            # Sample weight - Gives more weight to samples with better (more negative) scores.
            train_sample_weight = np.array(
                [((i / self.XYs[:, -2].max()) ** 5) for i in train_data[:, -2]]).squeeze()

            # fit model:
            self.model.fit(train_data.iloc[:, :-3], train_data.iloc[:, -3], sample_weight = train_sample_weight)

        else:
            self.model.fit(train_data.iloc[:, :-3], train_data.iloc[:, -3])

        pred_train, results_train = self.evaluate(scaled_x_data = train_data.iloc[:,:-3], y_true = train_data.iloc[:,-2])
        pred_valid, results_valid = self.evaluate(scaled_x_data = valid_data.iloc[:,:-3], y_true = valid_data.iloc[:,-2])

        if self.show_plots == True:

            # shap_force_plot:
            self.shap_plot(X_train = train_data.iloc[:, :-3], Y_train=train_data.iloc[:, -3])

            # Print residuals plot:
            self.plot_residuals (model_name = 'Model5_Lasso Regression',
                         y_true_valid= valid_data.iloc[:, -2], valid_pred = pred_valid, cv=cv_round,
                         mse_valid_full=results_valid['MSE'], mse_valid_below_threshold=results_valid['MSE_threshold'])

            # Print expected vs. observed plot:
            self.plot_ys(model_name = 'Model5_Lasso Regression', y_true_train=train_data.iloc [:, -2], train_pred=pred_train,
                         y_true_valid=valid_data.iloc[:, -2], valid_pred=pred_valid, cv=cv_round,
                         mse_valid_full=results_valid['MSE'], mse_valid_below_threshold=results_valid['MSE_threshold'])

        return results_train, results_valid

    def features_report(self, X_train, Y_train, global_shap_values):
        data = X_train.copy()
        data['max_affin'] = Y_train
        correlation_map = data.corr()
        # correlation_map.to_excel("Hit_map_model14.xlsx")
        self.features_analysis = pd.DataFrame(index=self.data.columns)
        self.features_analysis['Features'] = self.data.columns.values
        self.features_analysis['Correlation_to_Y'] = correlation_map.iloc[-1]
        self.features_analysis['Correlation_to_Y_abs'] = abs(self.features_analysis['Correlation_to_Y'])
        # self.features_analysis.to_excel("Hit_map_model5.xlsx")

        self.features_analysis = self.features_analysis.iloc[:-1]
        self.features_analysis['Global_Shap_Value'] = global_shap_values
        self.features_analysis['Coefficients'] = self.model.coef_
        self.features_analysis['Coefficients_abs'] = abs(self.features_analysis['Coefficients'] )

        # Features analysis:
        self.features_analysis = self.features_analysis.sort_values(by=['Coefficients_abs'], ascending=False)  # sort by shap values

        # Shap values Plot:
        sns.set_style("darkgrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 27), sharey=True, sharex=False)
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

        # Dependence plot (can insert any i. Rank 0 = the best i):
        # A dependence plot is a scatter plot that shows the effect a single i has on the predictions made by the model.
        for i in range (4):
            plt.figure()
            shap.dependence_plot(f"rank({i})", shap_values, X_train, interaction_index=None, show=False)
            plt.savefig(f"Feature_{i}_dependence_plot.png", bbox_inches='tight', dpi=600)
            plt.show()

    def pca (self, n_components, train_data, valid_data):
        X_train = train_data.iloc[:, :-3]
        X_valid = valid_data.iloc[:, :-3]
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_valid_pca = pca.transform(X_valid)
        print(f"original x train shape: {X_train.shape}")
        print(f"transformed x train shape: {X_train_pca.shape}")
        train_pca_data = np.concatenate((X_train_pca, train_data.iloc[:, -3:]), axis=1)
        valid_pca_data = np.concatenate((X_valid_pca, valid_data.iloc[:, -3:]), axis=1)

        return pd.DataFrame(train_pca_data), pd.DataFrame(valid_pca_data)

    def feature_importance_rfe(self, train_data, valid_data, best_n_features):

        model = Ridge (alpha = 0.2)
        rfe = RFE(model, best_n_features)
        fit = rfe.fit(train_data.iloc[:, :-3], train_data.iloc[:, -3])  # train the model

        names = list(self.data.columns)
        print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

        selected_features = fit.support_
        feature_ranking = fit.ranking_

        best_features_index = np.where(selected_features)[0]
        X_train_best_features = train_data.iloc[:,best_features_index]
        X_valid_best_features = valid_data.iloc[:, best_features_index]
        train_data = np.concatenate((X_train_best_features, train_data.iloc[:,-3:]), axis=1)
        valid_data = np.concatenate((X_valid_best_features, valid_data.iloc[:,-3:]), axis=1)
        return pd.DataFrame(train_data), pd.DataFrame(valid_data), selected_features,  feature_ranking


    def predict (self, data_to_predict):

        index_list = [x for x in data_to_predict.columns if x in self.final_test_data.columns]
        data_to_predict = data_to_predict[index_list]

        # Scale the new data
        scaled_x_data = self.scalar_x.transform(data_to_predict)

        # Predict with train model:
        scaled_pred_using_train_data = self.model.predict(scaled_x_data)
        scaled_pred_using_train_data = np.array(scaled_pred_using_train_data).reshape(-1, 1)
        pred_using_train_data = pd.DataFrame(self.scalar_y.inverse_transform(scaled_pred_using_train_data), index=[data_to_predict.index.values])

        ## Predict with all data model:
        # train_test_data = pd.concat([self.data, self.final_test_data])
        # results_train, results = self.fit_model(train_data = train_test_data, valid_data = train_test_data, sample_weights=False, smote=smote,cv_round=1, alpha=alpha, y_cor_threshold=y_cor_threshold,x_cor_threshold=x_cor_threshold)
        # pred_using_all_data = self.model.predict(scaled_x_data)
        # pred = pd.DataFrame(np.c_[pred_using_all_data, pred_using_train_data], index = [data_to_predict.index.values], columns = ['Pred_100%', 'Pred_80%'])

        return pred_using_train_data

    def evaluate (self, scaled_x_data, y_true):
        scaled_pred = self.model.predict(scaled_x_data)
        scaled_pred = np.array(scaled_pred).reshape(-1,1)
        pred = self.scalar_y.inverse_transform(scaled_pred)

        MSE_full = mean_squared_error(y_true, pred)
        MAE_full = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)

        k = scaled_x_data.shape[1]
        n = len(y_true)

        adjusted_r2 = 1 - (((1 - r2) * (n - 1)) / (n - k - 1))
        tss = ((y_true - y_true.mean())**2).sum()
        rss = ((y_true - pred.squeeze())**2).sum()
        F_statics = ((tss-rss)/k)/(rss/(n-k-1))


        if self.CV_Kfolds == 'loo':
            return pred, MSE_full, MAE_full, 0, 0, 0

        else:
            # Calculate validation MSE to the best scores only:
            ys = pd.DataFrame({'y_true': y_true, 'y_pred': pred.squeeze()})                    # Create a new list with the best y_true scores (below threshold)
            ys = ys.sort_values(by=['y_true'])                                                 # sort scores by y_true
            ys = ys.loc[ys.y_true <= self.threshold_value]                                     # trim list by y_true threshold
            MSE_best_scores = mean_squared_error(ys.y_true, ys.y_pred)                         # calaculate MSE for best scores
            MAE_best_scores = mean_absolute_error (ys.y_true, ys.y_pred)                       # calaculate MAE for best scores

            results = pd.Series([MSE_full, MSE_best_scores, MAE_full, MAE_best_scores, r2, adjusted_r2,  F_statics], index=['MSE', 'MSE_threshold', 'MAE', 'MAE_threshold', 'R2', 'Adj_R2', 'F_statics'])

            return pred, results

    def plot_residuals(self, model_name, y_true_valid, valid_pred, cv, mse_valid_full, mse_valid_below_threshold):

        residuals = y_true_valid - valid_pred.squeeze()

        plt.scatter(valid_pred.squeeze(), residuals, color='black', s=7)
        plt.title(f"{model_name}, {self.title}, CV {cv}\n MSE full {round(mse_valid_full, 3)}, MSE below threshold {round(mse_valid_below_threshold, 3)}")
        plt.xlabel("y_true predicted")
        plt.ylabel("Residuals (y_true true - y_true predicted)")
        plt.grid()
        if self.save_plots == True:
            plt.savefig(f'N_Residuals_plot_{self.title}_cv_{cv}.png')
        plt.show()
        print('Plot is ready')
        return self.residuals

    def plot_ys(self, model_name, y_true_train, train_pred, y_true_valid, valid_pred, cv, mse_valid_full, mse_valid_below_threshold):
        # Plot predictions vs. Y_true

        linreg = LinearRegression().fit(valid_pred, y_true_valid)
        x = np.linspace(-5, -15, 20)
        plt.plot(x, x , '--',color='gray')
        plt.plot(x, linreg.coef_ * x + linreg.intercept_, color='red')
        #plt.scatter(train_pred, y_true_train, color='blue', s=6)
        plt.scatter(valid_pred, y_true_valid, color='black', s=6)
        #plt.legend(['R^2 = 1','predicted', 'Train','validation'], loc='upper left')
        plt.legend(['Perfect', 'Predicted', 'Validation'], loc='upper left')
        plt.title(f"{model_name}, {self.title}, CV {cv}\n MSE full {round(mse_valid_full,3)}, MSE below threshold {round(mse_valid_below_threshold,3)}")
        plt.xlabel("y_true predicted")
        plt.ylabel("y_true true")
        plt.grid()
        if self.save_plots == True:
            plt.savefig(f'N_Ys_plot_{self.title}_cv_{cv}.png')
        plt.show()
        print ('Plot is ready')

    def save_model(self, file_name):
        # save_plots the model:
        filename = f'Saved_model_{file_name}.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        print ('Model have been saved')
        # save_plots the scalar:
        pickle.dump(self.scalar_x, open(f'Saved_scalar_x_{file_name}.sav', 'wb'))
        pickle.dump(self.scalar_y, open(f'Saved_scalar_y_{file_name}.sav', 'wb'))
        print('Scalar have been saved')
        # Save features names:
        #features = list(self.final_test_data.columns)
        pd.Series(self.final_test_data.columns).to_csv(r'C:\Users\froma\Desktop\ML\Project Korona\Lihi\Saved_features_names_model_5.csv')
        print ('Features have been saved')

#######################################33
def run_over_models(cv=10, scaler = 'Standard', scale_by=(-100,100), smote = False, sample_weights=False ):
    RF = RandomForestRegressor(max_depth=10, max_features=24, random_state=1)

    models_names = [ 'Ridge Regression','Model5_Lasso Regression', 'Gradient Boosting ls', 'Gradient Boosting lad','Neural Network', 'XGBoost','AdaBoostRegressor_linear','AdaBoostRegressor_sequred',  'DecisionTree', 'Adaboost-RF',  'BayesianRidge',  'SVR','RF mse']
    models_list = [ Ridge ( alpha = 0.2, max_iter=40000, random_state=1), Lasso(alpha = 1e-5, max_iter=20000), GradientBoostingRegressor( loss = 'ls',learning_rate=0.3, max_depth=3, n_estimators=100, random_state=1), GradientBoostingRegressor( loss = 'lad',learning_rate=0.3, max_depth=3, n_estimators=100, random_state=1),MLPRegressor(learning_rate_init=0.005, max_iter=500, batch_size=64, random_state=1), xgb.XGBRegressor(objective="reg:squarederror", learning_rate = 0.2,  max_depth = 4, n_estimators = 100, random_state=1, n_jobs=-1),AdaBoostRegressor(learning_rate=0.01, loss ='linear', random_state=1),AdaBoostRegressor(learning_rate=0.01, loss ='square', random_state=1),  tree.DecisionTreeRegressor(random_state=1, max_depth=6), AdaBoostRegressor(base_estimator=RF, learning_rate=1, loss='linear', n_estimators=50,random_state=2),  BayesianRidge(),  svm.SVR(),RandomForestRegressor(criterion = 'mse',random_state=1)]
    models_list = pd.Series(models_list, index=models_names)

    MSE_cv_results = pd.DataFrame(index=['MSE_full', 'MSE_threshold', 'MSE_variance', 'MAE_full', 'MAE_threshold','MAE_variance', 'r2'])

    results_table = pd.DataFrame()
    for i in range(len(models_list)):

        train_model = Model(train_data, threshold_percent=10, random_state=1)
        MSEs_train, MSEs_test = train_model.main(model_name=models_names[i], scaler = scaler,model=models_list.iloc[i],title=models_names[i], scale_by=scale_by, CV_Kfolds=cv,
                                   sample_weights=sample_weights, smote=smote,  pca_components=False, keep_best_features=False)

        results_table = results_table.append(pd.Series(MSEs_test, name= models_names[i], index=['MSE','MSE_threshold','MSE_Variance', 'MAE','MAE_threshold', 'MAE_Variance','R2' ]))


    table = lambda MSE_cv_results: tabulate(results_table, headers='keys', tablefmt='psql')
    print(table(MSE_cv_results))

    return results_table
############################### RD_Features - hyper parametrs tuning #############################################################################################33

if __name__ == '__main__':

    #Open data_origin:

    train_data = pd.read_csv(r"C:\Users\froma\Desktop\ML\Project Korona\Lihi\model15\program_files\Train_data_m15__13_7_20.csv", index_col=0)
    final_test_RD_Data = pd.read_csv(r"C:\Users\froma\Desktop\ML\Project Korona\Lihi\model15\program_files\Test_data_m15__13_7_20.csv", index_col=0)


########################################################################################3
    print ('scale 100')
    results_table = run_over_models( cv=10, scale_by=100, smote=False, sample_weights=False)




