import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm


class LinearRegressionModel:
    def __init__(self, file_path, feature_selection=False, 
                scaler=None,
                drop_columns=None,
                outlier_filter=True,
                collinear_features=True,
                high_leverage_points=True):
        self.file_path = file_path
        self.model = LinearRegression()
        self.feature_selection = feature_selection
        self.selector = None
        self.scaler = scaler if scaler else None
        self.outlier_filter = outlier_filter
        self.collinear_features = collinear_features
        self.high_leverage_points = high_leverage_points
        self.load_data(drop_columns=drop_columns)
    
    def remove_high_leverage_points(self, threshold=3):
        """Remove the high leverage points from the data.

        Args:
            threshold (int, optional): Threshold value for the high leverage points. Defaults to 3.
        """
        # Fit a linear regression model
        model = sm.OLS(self.data.iloc[:, -1], sm.add_constant(self.data.iloc[:, :-1])).fit()

        # Calculate leverage
        infl = OLSInfluence(model)
        leverage = infl.hat_matrix_diag

        # Remove high leverage points
        print('Number of removed high leverage points:', len(self.data[leverage > threshold]))
        self.data = self.data[leverage <= threshold]

        return self.data
    
    def vif(self, dataframe, threshold=10.0):
        """
        Variance Inflation Factor (VIF) is a measure of multicollinearity among the independent variables within a multiple regression model.
        It is calculated by taking the the ratio of the variance of all a given model's betas divide by the variane of a single beta if it were fit alone.
        """
        columns = dataframe.columns
        vif_series = pd.Series()
        for col_name in columns:
            X = dataframe.drop(columns = [col_name]).values
            y = dataframe[col_name].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r_sq = reg.score(X, y)
            vif_series[col_name] = 1/(1 - r_sq)
        # Get the collinear features based on the threshold
        print('Variance Inflation Factor (VIF):', vif_series)
        collinear_features = vif_series[vif_series > threshold].index.values
        return collinear_features
    
    def combine_collinear_features(self, threshold=10.0):
        """Combine the collinear features based on the threshold.

        Args:
            threshold (float, optional): Threshold value for the collinear features. Defaults to 10.0.
        """
        collinear_features = self.vif(self.data, threshold=threshold)
        if len(collinear_features) == 0:
            return
        # only take n-1 collinear features to avoid multicollinearity
        if len(collinear_features) > 1:
            collinear_features = collinear_features[:-1]
        for i in range(len(collinear_features)):
            for j in range(i+1, len(collinear_features)):
                self.data[f'{collinear_features[i]}_{collinear_features[j]}'] = self.data[collinear_features[i]] * self.data[collinear_features[j]]
        self.data = self.data.drop(columns=collinear_features)

        
    def filter_outliers(self):
        """Filter out the outliers using IQR method.
        """
        for column in self.data:
            if self.data[column].dtype in ['int64', 'float64']:
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.data[(self.data[column] < (Q1 - 1.5 * IQR)) | (self.data[column] > (Q3 + 1.5 * IQR))]
                # Filter out the outliers
                self.data = self.data[(self.data[column] >= (Q1 - 1.5 * IQR)) & (self.data[column] <= (Q3 + 1.5 * IQR))]

    def search_best_number_of_features(self, max_features=9):
        """Search for the best number of features to use in the model.

        Args:
            max_features (int, optional): Number of features in the data. Defaults to 9.

        Returns:
            best_n_feature: The best number of features to use in the model.
        """
        best_rmse = float('inf')
        best_n_features = 0
        for n_features in range(1, max_features+1):
            rmse = self.train(n_features=n_features)
            if rmse < best_rmse:
                best_rmse = rmse
                best_n_features = n_features
        print(f"Best number of features: {best_n_features}, RMSE: {best_rmse}")
        return best_n_features
    
    def perform_feature_selection(self, n_features=9):
        """Perform feature selection using SelectKBest.

        Args:
            n_features (int, optional): Number of features to be selected. Defaults to 9.
        """
        self.selector = SelectKBest(score_func=f_regression, k=n_features)
        self.X_transformed = pd.DataFrame(self.selector.fit_transform(self.X, self.y))
        return self.selector.get_feature_names_out()
    
    def load_data(self, drop_columns=None):
        """Load the data from the file and filter out the outliers.
        """
        self.data = pd.read_csv(self.file_path)
        if drop_columns:
            self.data = self.data.drop(columns=drop_columns)
        # Drop the first for index and last columns for target
        self.data = self.data.iloc[:, 1:]
        print('Number of rows before filtering:', len(self.data))
        if self.outlier_filter:
            self.filter_outliers()
        if self.high_leverage_points:
            self.remove_high_leverage_points()
        if self.collinear_features:
            self.combine_collinear_features()
        print('Number of rows after filtering:', len(self.data))
        self.X = self.data.drop(columns=['Y'])
        self.y = self.data['Y'] 
        if self.scaler:
            self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)
    
    def train(self, n_features):
        """Train the model using the data.

        Args:
            n_features (int): Number of features to use in the model.

        Returns:
            avg_rmse: Average RMSE of the model.
        """
        if self.feature_selection:
            selected_features = self.perform_feature_selection(n_features=n_features)
        rmse_scores = []
        r2_scores = []
        features = self.X if not self.feature_selection else self.X_transformed
        print(f'Training with {features.shape[1]} features', selected_features)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []
        for train_index, test_index in kf.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            rmse_scores.append(root_mean_squared_error(y_test, predictions))
            r2_scores.append(r2_score(y_test, predictions))
        avg_rmse = sum(rmse_scores) / len(rmse_scores)
        avg_r2 = sum(r2_scores) / len(r2_scores)
        print(f'Training completed. Average RMSE: {avg_rmse}, Average R2: {avg_r2}')
        return avg_rmse

    def predict(self, input_data):
        """Predict the output using the input data.

        Args:
            input_data (DataFrame): Input data to predict the output.

        Returns:
            output: Predicted output.
        """
        if self.scaler:
            input_data = pd.DataFrame(self.scaler.transform(input_data), columns=input_data.columns)
        return self.model.predict(input_data)
    
    def test(self, test_file_path, drop_columns=None):
        """Test the model using the test data.

        Args:
            test_file_path (str): csv file path for the test data.

        Returns:
            predictions: Predictions for the test data.
        """
        test_data = pd.read_csv(test_file_path)
        if drop_columns:
            test_data = test_data.drop(columns=drop_columns)
        X_test = test_data.iloc[:, 1:]
        if self.scaler:
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
        if self.feature_selection:
            X_test = self.selector.transform(X_test)
        predictions = self.model.predict(X_test)
        return predictions
    
    def plot_predictions(self, x_test, y_test, predictions):
        """Plot the predictions against the actual values.

        Args:
            x_test (Series): X values of the test data.
            y_test (Series): Y values of the test data.
            predictions (Series): Predicted values.
        """
        for i in range(x_test.shape[-1]):
            x = x_test.iloc[:, i]
            plt.scatter(x, y_test, color='gray')
            plt.scatter(x, predictions, color='red', linewidth=2)
            plt.show()
    
class KNNModel(LinearRegressionModel):
    def __init__(self, file_path, 
                feature_selection=False, 
                scaler=None,
                drop_columns=None,
                outlier_filter=True,
                collinear_features=True,
                high_leverage_points=True,
                n_neighbors=5):
        super().__init__(file_path, 
                        feature_selection=feature_selection, 
                        scaler=scaler,
                        drop_columns=drop_columns,
                        outlier_filter=outlier_filter,
                        collinear_features=collinear_features,
                        high_leverage_points=high_leverage_points)
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)


if __name__ == '__main__':
    # Usage
    drop_columns = None
    print('========================= Linear Regression Model =========================')
    lr = LinearRegressionModel('train_ch.csv', 
                            feature_selection=True, 
                            scaler=StandardScaler(),
                            drop_columns=drop_columns,
                            outlier_filter=True,
                            collinear_features=False,
                            high_leverage_points=False)
    best_n_features = lr.search_best_number_of_features(max_features=9)
    lr.train(n_features=best_n_features)
    # predicts = lr.test('test_ch.csv')
    # print(predicts)
    # print('========================= KNN Model =========================')
    # knn = KNNModel('train_ch.csv', feature_selection=True)
    # best_n_features = knn.search_best_number_of_features(max_features=9)
    # knn.train(n_features=best_n_features)