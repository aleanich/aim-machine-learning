from aim_machine_learning.base_regressor import Regressor
from aim_machine_learning.neighbor_regressor import NeighborRegressor
from aim_machine_learning.metrics import Evaluator
import numpy as np
from itertools import islice

class ModelEvaluator:
    def __init__(self,model_class,params,X,y):
        self.k=params['k']
        self.model_class=model_class
        self.model=model_class(self.k)
        self.X=X
        self.y=y
    
    def train_test_split_eval(self,eval_obj,test_proportion):
        if not issubclass(self.model_class,Regressor):
            raise NameError('Deve essere figlio di regressor.')
        if not isinstance(eval_obj,Evaluator):
            raise NameError('Deve essere figlio di Evaluator.')

        n=int(self.X.shape[0]*test_proportion)
        X_test=self.X[0:n,:]
        y_train=self.y[n:]
        X_train=self.X[n:,:]

        self.model.fit(X_train,y_train)
        return self.model.evaluate(X_test,self.y[0:n][:],eval_obj)

    def kfold_cv_eval(self,eval_obj,K):
        n=int(self.X.shape[0]/K)
        err=np.zeros(K)
        std=np.zeros(K)
        for i in range(K):
            if i>0 & i<K-1:
                X_train1=self.X[0:n*i,:]
                X_train2=self.X[n*(i+1):,:]
                X_train=np.vstack((X_train1,X_train2))
                y_train1=self.y[0:n*i]
                y_train2=self.y[n*(i+1):]
                y_train=np.hstack((y_train1,y_train2))
                X_test=self.X[i*n:(i+1)*n,:]
                y_test=self.y[i*n:(i+1)*n]
            if i==K-1:
                X_train=self.X[0:(K-1)*n,:]
                y_train=self.y[0:(K-1)*n]
                y_test=self.y[(K-1)*n:]
                X_test=self.X[(K-1)*n:,:]
            if i==0:
                X_train=self.X[n:,:]
                y_train=self.y[n:]
                y_test=self.y[0:n]
                X_test=self.X[0:n,:]
            
            self.model.fit(X_train,y_train)
            val=self.model.evaluate(X_test,y_test,eval_obj)
            if eval_obj.current_metric=='corr':
                err[i]=val['corr']
            if eval_obj.current_metric=='mse':
                err[i]=val['mean']
                std[i]=val['std']
        if eval_obj.current_metric=='corr':
            return {'corr':np.mean(err)}
        if eval_obj.current_metric=='mse':
            return {'mean':np.mean(err),'std':np.mean(std)}
            


        

                



            
        






   