from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np

class KNNClassifier:
    knnclassifier = KNeighborsClassifier()

    def LoadDataset(self):
        print('Loading dataset')
        self.dataset = datasets.load_iris()
        self.data = pd.DataFrame(self.dataset.data,columns=self.dataset.feature_names)
        self.target = pd.DataFrame(self.dataset.target,columns=['setosa'])
        print(self.target.shape)

    def SplitDataset(self):
        print('Splitting Dataset')
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.data,self.target,test_size=0.7,random_state=0)
        print(self.y_train.shape)

    def TrainModel(self):
        print('Training Model')
        self.knnclassifier.fit(self.x_train,self.y_train.values.ravel())

    def Predict(self):
        print('Predicting')
        self.y_pred = self.knnclassifier.predict(self.x_test)

    def TestModel(self):
        print('Accuracy: ',metrics.accuracy_score(self.y_test,self.y_pred))
        print('Precision: ',metrics.precision_score(self.y_test,self.y_pred,average='macro'))
        print('F1 Score: ',metrics.f1_score(self.y_test,self.y_pred,average='macro'))
        print('Recall: ',metrics.recall_score(self.y_test, self.y_pred, average='macro'))



def main():
    knn = KNNClassifier()
    knn.LoadDataset()
    knn.SplitDataset()
    knn.TrainModel()
    knn.Predict()
    knn.TestModel()

main()
