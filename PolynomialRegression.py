from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

class PolynomialRegression:
    lm = linear_model.LinearRegression()

    def LoadDataset(self):
        print('Loading dataset')
        self.data = datasets.load_boston()
        self.dataframe = pd.DataFrame(self.data.data,columns=self.data.feature_names)
        self.target = pd.DataFrame(self.data.target,columns=['MEDV'])
        self.x = self.dataframe
        self.y = self.target["MEDV"]

    def SplitDataset(self):
        print('Splitting dataset')
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.3,random_state=5)
        polynomialFeatures = PolynomialFeatures(degree=2)
        self.x_train_poly = polynomialFeatures.fit_transform(self.x_train)
        self.x_test_poly = polynomialFeatures.fit_transform(self.x_test)

    def TrainModel(self):
        print('Training model')
        self.lm.fit(self.x_train_poly,self.y_train)

    def Predict(self):
        print('Predicting output')
        self.y_predicted = self.lm.predict(self.x_test_poly)

    def TestModel(self):
        print('R2 Score: ',metrics.r2_score(self.y_test,self.y_predicted))
        print('Explained Variance Score:', metrics.explained_variance_score(self.y_test, self.y_predicted))
        print('Mean Squared Error:', metrics.mean_squared_error(self.y_test, self.y_predicted))


def main():
    pr = PolynomialRegression()
    pr.LoadDataset()
    pr.SplitDataset()
    pr.TrainModel()
    pr.Predict()
    pr.TestModel()

main()