from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np

class DecisionTreeRegression:
    regressor = DecisionTreeRegressor(random_state=0)

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

    def TrainModel(self):
        print('Training model')
        self.regressor.fit(self.x_train,self.y_train)

    def Predict(self):
        print('Predicting output')
        self.y_predicted = self.regressor.predict(self.x_test)

    def TestModel(self):
        print('R2 Score: ',metrics.r2_score(self.y_test,self.y_predicted))
        print('Mean Squared Error:', metrics.mean_squared_error(self.y_test, self.y_predicted))


def main():
    dtr = DecisionTreeRegression()
    dtr.LoadDataset()
    dtr.SplitDataset()
    dtr.TrainModel()
    dtr.Predict()
    dtr.TestModel()

main()