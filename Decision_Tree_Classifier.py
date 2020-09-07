# By Lihi Gur-Arie

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, f1_score
from tqdm import tqdm
from sklearn.tree import plot_tree


class TreesClasiffier:
    """
    Decision Tree Classifier.
    If X_test & Y_test are not provided, evaluation will be on the mean cross validation.
    If test data provided, the model will run once over all the training data and will be evaluated on the test data.

    Parameters
    ----------
    X_train  : pd.DataFrame of shape (n_samples, n_features)
        Training data
    Y_train  : pd.Series
        Training Label
    X_test  : pd.DataFrame of shape (n_samples, n_features)
        Test data
    Y_test  : pd.Series
        Test Label
    verbose : int
    """

    def __init__(self, X_train, Y_train, X_test=None, Y_test=None, verbose=1):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.verbose = verbose

    def main(self, cv=10, max_depth=3, show_plots=False, save_plots=False):
        """
        Parameters
        ----------
        cv  : int
            The k-fold amount for cross validation
        max_depth  : int
            The maximal depth of the tree
        show_plots  : bool
            If True, plot will be added
        save_plots  : bool
            If True, plot will be saved

        Returns
        -------
        results_valid : pd.DataFrame
            A table containing the validation evaluation.
        """
        self.max_depth = max_depth
        self.cv = cv
        self.save_plots = save_plots
        self.show_plots = show_plots

        # If test data is not supplied, preform cross validation:
        if self.X_test is None:
            results_valid = self.cvKfold()

        # If there is supplied, run over all the training group without cross-validation:
        else:
            results_valid = self.fit_model(X_train=self.X_train, y_train=self.Y_train, X_valid=self.X_test,
                                           y_valid=self.Y_test, cv_round=0)

        return results_valid

    def cvKfold(self):
        # Split the data into train / validation in a stratified way.
        # The function returns the average results of all the k folds.

        cv_round = 1  # count the round of the cv

        results_valid = pd.DataFrame()
        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=1)

        for train_index, valid_index in splitter.split(self.X_train, self.Y_train):
            if self.verbose == 1:
                print(colored(f'CV {cv_round}', 'cyan'))

            validation_results = self.fit_model(X_train=self.X_train.iloc[train_index],
                                                y_train=self.Y_train.iloc[train_index],
                                                X_valid=self.X_train.iloc[valid_index],
                                                y_valid=self.Y_train.iloc[valid_index], cv_round=cv_round)

            results_valid[f'CV {cv_round}'] = validation_results
            cv_round += 1

        return results_valid.mean(axis=1)


    def fit_model(self, X_train, y_train, X_valid, y_valid, cv_round, min_samples_split=2):

        model = DecisionTreeClassifier(class_weight='balanced', max_depth=self.max_depth,
                                       min_samples_split=min_samples_split, random_state=1).fit(X_train, y_train)
        undr = model.tree_
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)

        # Evaluate:
        recall_positive_train = recall_score(y_train, y_pred_train)
        recall_positive_valid = recall_score(y_valid, y_pred_valid)
        precision_positive_train = precision_score(y_train, y_pred_train)
        precision_positive_valid = precision_score(y_valid, y_pred_valid)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_valid = accuracy_score(y_valid, y_pred_valid)
        f1_train = f1_score(y_train, y_pred_train)
        f1_valid = f1_score(y_valid, y_pred_valid)

        # Print results
        if self.verbose == 1:
            print(
                f'\nValidation:\nrecall_pos {recall_positive_valid},\nprecision_pos {round(precision_positive_valid, 2)},\nf1_score {round(f1_valid, 2)}\naccuracy {round(accuracy_valid, 2)}\n')
            print(colored('Test confusion matrix:', 'green'))
            confusion_matrix_valid = pd.DataFrame(confusion_matrix(y_valid, y_pred_valid),
                                                  columns=['Predicted Not Binds', 'Predicted Binds'],
                                                  index=['True Not Binds', 'True Binds'])
            print(confusion_matrix_valid)

        results = pd.Series(
            [recall_positive_valid, precision_positive_valid, f1_valid, accuracy_valid, recall_positive_train,
             precision_positive_train, f1_train, accuracy_train], name=cv_round,
            index=['recall_valid', 'precision_valid', 'f1_score_valid', 'accuracy_valid', 'recall_train',
                   'precision_train', 'f1_score_train', 'accuracy_train'])

        # Create a decision Tree plot:
        if self.show_plots == True:
            plt.figure(figsize=(16, 12))
            plot_tree(model, feature_names=X_train.columns.values, fontsize=12, filled=True,
                          class_names=['No_Binding', 'Binding'])

            # Save the plot:
            if self.save_plots == True:
                plt.savefig(f"Decision_Tree_plot.png")

            plt.show()
            print('Plot is ready')

        return results

##################################################################################3

class Remove_features_by_Forward_feature_addition_Tree:
    """
    Feature reduction by forward step.

    Parameters
    ----------
    X  : pd.DataFrame of shape (n_samples, n_features)
        Training data
    Y  : pd.Series
        Training Label
    """

    def __init__(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()

    def main(self, max_depth, cv):
        """
        Parameters
        ----------
        max_depth  : int
            The maximal depth of the tree
        cv  : int
            The k-fold amount for cross validation
        """
        self.max_depth = max_depth
        self.cv = cv

        # Select the best features:
        full_results_table, selected_features_table = self.reduce_features()

        # Save the relevant features:
        best_feature_recall = selected_features_table['recall'].copy().iloc[::-1].idxmax(axis=1, skipna=True)           # Get the name of the feature with the highest recall
        best_recall_results = selected_features_table.loc[ :best_feature_recall]                                        # Get all features results from the begining to the best recall feature
        best_feature_recall_and_precision = best_recall_results['precision'].copy().idxmax(axis=1,skipna=True)          # Get the name of the feature with the best precision
        best_features = selected_features_table.loc[ :best_feature_recall_and_precision].index.values                   # Get the results of the the feature with the best precision and recall
        best_results = selected_features_table.loc[best_feature_recall_and_precision]
        X_train = self.X[best_features.squeeze()]                                                                       # Get X_train with the best features combination
        features_number = X_train.shape[1]
        best_results['n_features'] = features_number
        print(f'Number of features before reduction = {self.X.shape[1]}')
        print(f'Number of features after reduction = {features_number}')

        return X_train, full_results_table, selected_features_table, best_results

    def reduce_features(self):
        full_results_table = pd.DataFrame()
        selected_features_table = pd.DataFrame()

        counter = 1
        stop_by_recall_counter = 0
        previous_recall = 0
        features_left_names = self.X.columns

        for i in tqdm(range(self.X.shape[1])):

            all_results, new_feature = self.scan_features(selected_features_names=selected_features_table.index.array,
                                                          counter=counter, features_left_names=features_left_names)
            all_results['Round'] = counter
            # Append the i with the best mse score:
            selected_features_table = selected_features_table.append(new_feature)
            full_results_table = full_results_table.append(all_results)

            # Early stoping, if the recall is getting worse:
            if new_feature['recall'] <= previous_recall:                                                                # if the current recall is worse then the previous, count
                stop_by_recall_counter += 1
                if stop_by_recall_counter == 5:
                    return full_results_table, selected_features_table

            # Remove features that are already chosen:
            features_left_names = features_left_names.drop(
                selected_features_table.index[-1])                                                                      # Append the i with the best Adj_r2 score
            counter += 1
        return full_results_table, selected_features_table

    def scan_features(self, selected_features_names, counter, features_left_names):

        results_table_valid = pd.DataFrame()

        for j in features_left_names:
            if counter == 1:
                current_X = pd.DataFrame(self.X[j])
            else:
                idx = np.concatenate((selected_features_names, j), axis=None)
                current_X = self.X[idx]

            results_valid = TreesClasiffier(X_train=current_X, Y_train=self.Y, verbose=0).main(cv=self.cv,
                                                                                               max_depth=self.max_depth)
            results_valid.name = j
            results_table_valid = results_table_valid.append(pd.DataFrame(results_valid).T)

        results_table_valid = results_table_valid.sort_values(['recall', 'precision'], ascending=[False, False])        # Sort results
        best_feature_results = pd.Series(results_table_valid.iloc[0], name=results_table_valid.index[0])                # Get the best results
        print(f'Best feature is {best_feature_results.name}')
        return results_table_valid, best_feature_results