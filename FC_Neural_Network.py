# By Lihi Gur Arie

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

#### ANN #########################################################################

class ANN:
    """
    Fully connected Neural network regrresor.

    Parameters
    ----------
    X  :{array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values
    """

    def __init__(self, X, Y):

        self.X = X
        self.Y = Y
        self.title = None
        self.Ys = None
        self.random_state = None
        self.scalar_x = None
        self.scalar_y = None
        self.X_scaled = None
        self.Y_scaled = None
        self.threshold_value = None
        self.XYs = None

    def main (self, title, threshold_percent, epochs, batch_size, learning_rate, CV_Kfolds, sample_weights, random_state, scale_by, smote):
        """
        Parameters
        ----------
        title : str

        threshold_percent : int
            this parameter sets the precent threshold for the molecular affinity

        epochs : int
             Maximum number of epochs

        batch_size : int

        learning_rate : float

        CV_Kfolds : int
            Number of k-fold for Cross-Validation.

        sample_weights : bool
            if True, The model will use sample weights

        random_state : int

        scale_by : tuple
            define the 'MinMaxScaler' scaling range

        smote : bool
            If True, the model will use smote algorithm to balance the data


        Returns
        -------
        results : pd.Series
            The results scores.
        """
        self.find_threshold_value(threshold_percent)
        self.title = title
        self.random_state = random_state
        self.scale_data(scale_by = scale_by)

        MSE_full_length, MSE_under_threshold,MSE_variance, MAE_full_length, MAE_under_threshold, MAE_variance, r2 = self.ann_fit(epochs=epochs, learning_rate = learning_rate, batch_size=batch_size, sample_weights=sample_weights, smote=smote, CV_Kfolds = CV_Kfolds)
        results = [MSE_full_length, MSE_under_threshold,MSE_variance, MAE_full_length, MAE_under_threshold, MAE_variance, r2]

        return pd.Series(results)

    def find_threshold_value (self, threshold_percent):
        """
        Determine the threshold of the best affinity scores

        Parameters
        ----------
        threshold_percent : int
        """

        threshold_index = int(self.Y.shape[0] * (threshold_percent / 100))
        self.threshold_value = self.Y.sort_values()[threshold_index]
        print (f'The {threshold_percent}% threshold is {self.threshold_value}')

    def scale_data (self, scale_by):
        """
        Scale the data using MinMax scaler

        Parameters
        ----------
        scale_by : tuple
            The range to scale the data to

        """
        # Scale each i individually between the desired values
        self.scalar_x = MinMaxScaler(feature_range=(-scale_by, scale_by))
        self.scalar_x.fit(self.X)
        self.X_scaled = self.scalar_x.transform(self.X)

        # Scale Y between the desired values
        self.Y = np.array(self.Y).reshape((-1, 1))
        self.scalar_y = MinMaxScaler(feature_range=(-scale_by, scale_by))
        self.scalar_y.fit(self.Y)
        self.Y_scaled = self.scalar_y.transform(self.Y)

        # Create a Data table with the features (X), the unscaled scores(y_true unscaled),
        # the scaled scores (y_true scaled) and the above/under threshold scores (y class)
        self.Ys = pd.DataFrame(self.Y_scaled)
        self.Ys['unscaled'] = self.Y
        self.Ys['class'] = np.array([1.0 if y > self.threshold_value else -1.0 for y in self.Y])
        self.Ys = np.array(self.Ys)
        self.XYs = np.concatenate((self.X_scaled, self.Ys), axis=1)

    def smote (self, XYs_train):
        """
        Use smote algorithm to compensate for the imbalanced data
        """

        # transform the dataset:
        X_smote_scaled, y_smote_scaled = SMOTE(random_state=self.random_state, n_jobs=-1).fit_resample(XYs_train[:,:-2], XYs_train[:,-1])

        X_train_scaled = X_smote_scaled[:, :-1]
        Y_train_scaled = X_smote_scaled[:, -1].reshape(-1, 1)
        Y = self.scalar_y.inverse_transform(Y_train_scaled)
        Y_class = (np.array([1.0 if y > self.threshold_value else -1.0 for y in Y])).reshape(-1, 1)
        XYs_smote = np.concatenate((X_train_scaled, Y_train_scaled, Y, Y_class), axis=1)
        return XYs_smote

    def ann_model (self, learning_rate):
        """
        Compile a 3 layers Neural network

        Parameters
        ----------
        learning_rate : float
        """

        # fix random seed for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        # model:
        model = Sequential()
        model.add(Dense(16, kernel_initializer='normal', use_bias=True, bias_initializer='zeros', input_dim=self.X.shape[1], activation='relu'))
        model.add(Dense(8, kernel_initializer='normal', use_bias=True, bias_initializer='zeros', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', use_bias=True, bias_initializer='zeros', activation='linear'))

        # Compile the network :
        opt = Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        return model

    def ann_fit (self, epochs, batch_size, learning_rate, sample_weights, CV_Kfolds, smote):
        """
        Fit the Neural network, for each cross validation
        """

        cv = 1                             # count the round of the cv
        print (f'CV {cv}')

        results_train = pd.DataFrame(index=['MSE_full', 'MSE_threshold','MAE_full', 'MAE_threshold', 'r^2_full' ])
        results_valid = pd.DataFrame(index=['MSE_full', 'MSE_threshold','MAE_full', 'MAE_threshold', 'r^2_full' ])

        kfold = StratifiedKFold(n_splits = CV_Kfolds, shuffle=True, random_state = self.random_state)

        for train_index, valid_index in kfold.split(self.XYs, self.Ys[:,-1]):

            model = self.ann_model (learning_rate = learning_rate)

            ###  Callbacks ###:
            current_time = datetime.datetime.now().strftime("%y_true%m%d-%H%M%S")
            logdir = r'ANN_TB_Lihi\28_4_20' + current_time

            # 1. checkpoint - save the network weights only when there is an improvement in the validation loss:
            filepath = logdir + "\weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

            # 2. EarlyStopping - Stop training when validation error has stopped improving:
            early = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=30, verbose=1, mode='auto')

            # 3. Tensorboard (tensorboard --logdir=):
            TB_callbacks = TensorBoard(log_dir=logdir, histogram_freq=1, write_images=True)

            current_train_XYs = self.XYs[train_index]

            if smote == True:
                current_train_XYs = self.smote(self.XYs[train_index])

            if sample_weights == True:
                # Sample weight - Gives more weight to samples with better (more negative) scores.
                train_sample_weight = np.array([((i / self.XYs[:, -2].max()) ** 5) for i in current_train_XYs[:, -2]]).squeeze()
                valid_sample_weight = np.array([((i / self.XYs[:, -2].max()) ** 5) for i in self.XYs[valid_index, -2]]).squeeze()
                print ('Sample weights are on')

                # fit model:
                history = model.fit(current_train_XYs[:,:-3], current_train_XYs[:,-3], epochs=epochs, batch_size = batch_size,
                                         validation_data=(self.XYs[valid_index,:-3], self.XYs[valid_index,-3], valid_sample_weight),
                                         sample_weight=train_sample_weight, verbose=0, callbacks=[checkpoint, early, TB_callbacks])

            else:
                print('Sample weights are off')
                history = model.fit(current_train_XYs[:,:-3], current_train_XYs[:,-3], epochs=epochs, batch_size = batch_size,
                                         validation_data=(self.XYs[valid_index,:-3], self.XYs[valid_index,-3]), verbose=0,
                                         callbacks=[checkpoint, early, TB_callbacks])

            # Plot epoch vs. loss for the train and validation
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'MSE loss - CV {cv}')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

            # Evaluate the model:
            pred_train, MSE_full_train, MSE_best_scores_train, MAE_full_train, MAE_best_scores_train, r2_full_train = self.evaluate(model = model, scaled_x_data = self.XYs[train_index, :-3], y_true=self.XYs[train_index, -2])
            pred_valid, MSE_full_valid, MSE_best_scores_valid, MAE_full_valid, MAE_best_scores_valid, r2_full_valid = self.evaluate(model = model, scaled_x_data = self.XYs[valid_index,:-3], y_true = self.XYs[valid_index,-2])

            results_train [f'CV {cv}'] = [MSE_full_train, MSE_best_scores_train, MAE_full_train, MAE_best_scores_train, r2_full_train]
            results_valid [f'CV {cv}'] = [MSE_full_valid, MSE_best_scores_valid, MAE_full_valid, MAE_best_scores_valid, r2_full_valid]

            print (f'CV {cv} Train:      full length MSE {MSE_full_train}, below threshold MSE  {MSE_best_scores_train}, R2 {r2_full_train}')
            print (f'CV {cv} Validation: full length MSE {MSE_full_valid}, below threshold MSE  {MSE_best_scores_valid}, R2 {r2_full_valid}')

            # Plot Y_true vs. Y_pred
            self.plot_ys( y_true_train = self.XYs[train_index, -2], train_pred = pred_train, y_true_valid = self.XYs[valid_index,-2], valid_pred = pred_valid, cv = cv, mse_valid_full = MSE_full_valid, mse_valid_below_threshold = MSE_best_scores_valid )
            cv += 1

        return results_valid.loc["MSE_full"].mean(), results_valid.loc['MSE_threshold'].mean(), results_valid.loc["MSE_full"].var(), results_valid.loc["MAE_full"].mean(), results_valid.loc['MAE_threshold'].mean(), results_valid.loc["MAE_full"].var(), results_valid.loc['r^2_full'].mean()

    def evaluate (self, model, scaled_x_data, y_true):
        """
        Evaluate model's predictions on the original data (without scaling)

        """
        scaled_pred = model.predict(scaled_x_data)
        pred = self.scalar_y.inverse_transform(scaled_pred)
        MSE_full = mean_squared_error(y_true, pred)
        MAE_full = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)

        # Calculate validation MSE to the best scores only:
        ys = pd.DataFrame({'y_true': y_true.squeeze(), 'y_pred': pred.squeeze()})                    # Create a new list with the best y_true scores (below threshold)
        ys.sort_values(by=['y_true'])                                                                # sort scores by y_true
        ys = ys[ys.y_true <= self.threshold_value]                                                   # trim list by y_true threshold
        MSE_best_scores = mean_squared_error(ys.y_true, ys.y_pred)
        MAE_best_scores = mean_absolute_error(ys.y_true, ys.y_pred)# claculate MSE for best scores
        print(f'MSE full length {MSE_full}, MSE low scores  {MSE_best_scores}, R^2 full length  {r2}')
        return pred, MSE_full, MSE_best_scores, MAE_full, MAE_best_scores, r2

    def plot_ys(self, y_true_train, train_pred, y_true_valid, valid_pred, cv, mse_valid_full, mse_valid_below_threshold):
        """
        Plot Y_pred vs. Y_true
        """
        linreg = LinearRegression().fit(valid_pred, y_true_valid)
        x = np.linspace(-5, -15, 20)
        plt.plot(x, x , '--',color='gray')
        plt.plot(x, linreg.coef_ * x + linreg.intercept_, color='red')
        plt.scatter(train_pred, y_true_train, color='blue', s=7)
        plt.scatter(valid_pred, y_true_valid, color='black', s=7)
        plt.legend(['R^2 = 1',f'predicted', 'Train','validation'], loc='upper left')
        plt.title(f"{self.title}, CV {cv}\n MSE full {round(mse_valid_full,3)}, MSE below threshold {round(mse_valid_below_threshold,3)}")
        plt.xlabel("y_true predicted")
        plt.ylabel("y_true true")
        plt.grid()
        plt.show()
        print ('Plot is ready')

####################################################################

class step_forward_feature_selection:
    """
    Remove features by step_forward.
    Features order are based on lowest MSE.
    """

    def __init__(self, data):
        self.data = data.copy()

    def main(self):

        # Select the best features:
        full_results_table, selected_features_table = self.reduce_features()

        # Save the relevant features:
        best_features = pd.Series(selected_features_table.index[:np.argmin(selected_features_table['MSE']) + 1])
        best_features[len(best_features)] = 'max_affin'         # Add 'max_affin' (y_true) to the features list

        print(f'Number of features before reduction = {self.data.shape[1]}')
        # Remove irrelevant features:
        train_data = self.data[best_features]
        print(f'Number of features after reduction = {train_data.shape[1]}')

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
            features_left_names = features_left_names.drop(selected_features_table.index[-1])     # Append the i with the best Adj_r2 score
            counter += 1
        return full_results_table, selected_features_table

    def scan_features(self, selected_features_names, counter, features_left_names):
        results_table_valid = pd.DataFrame()
        i=0
        for j in features_left_names:

            if counter == 1:
                current_data = self.data[[ j, 'max_affin']]
            else:
                idx = np.concatenate((selected_features_names, j, 'max_affin'), axis=None)
                current_data = self.data[idx]

            model = ANN(current_data, self.data.iloc[:, -1])
            results_valid = model.main(title='m15', epochs=1000, batch_size=64, learning_rate=0.0005, sample_weights=False, random_state=1, threshold_percent=10, scale_by=100, smote=False, CV_Kfolds=10)

            results_valid.name = j
            results_table_valid = results_table_valid.append(pd.DataFrame(results_valid).T)

        best_feature_index = np.argmin(results_table_valid['MSE'])                                             # The index of the i with the best mse
        best_feature_results = (results_table_valid.iloc[best_feature_index])                                  # Get the results of the best i
        best_feature_results = best_feature_results.rename (results_table_valid.index[best_feature_index])     # Give Name to the i
        print (f'Best feature is {best_feature_results.name}')
        return results_table_valid, best_feature_results



