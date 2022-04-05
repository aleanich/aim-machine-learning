from aim_machine_learning.base_regressor import Regressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from math import dist

class NeighborRegressor(Regressor):

    def __init__(self,k=1):
        self.k=k
    
    def fit(self,X,y):
        self.X=X
        self.y=y
    
    def IndiceDistMinima(self,x):
        y_pred=0
        distOld=-1
        for i in range(self.k):
            indMin=0
            distMin=dist(x,self.X[0][:])
            for j in range(self.X.shape[0]):
                if distMin>dist(x,self.X[j][:]):
                    if dist(x,self.X[j][:])>distOld:
                        indMin=j
                        distMin=dist(x,self.X[j][:])
            distOld=distMin
            y_pred=y_pred+self.y[indMin]
        return y_pred/self.k

    def predict(self,X):
        y_pred=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i]=self.IndiceDistMinima(X[i][:])
        return y_pred

#class MySklearnNeighborRegressor(KNeighborsRegressor):
    
    #def evaluate(self, X, y, eval_obj)