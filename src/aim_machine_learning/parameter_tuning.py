from aim_machine_learning.model_evaluator import ModelEvaluator
from aim_machine_learning.neighbor_regressor import NeighborRegressor
from aim_machine_learning.metrics import Evaluator
import matplotlib.pyplot as plt
import numpy as np

class ParametersTuner(ModelEvaluator):
    def __init__(self,model_class,X,y,supported_eval_types,output_path='output/'):
        self.model_class=model_class
        self.X=X
        self.y=y
        self.supported_eval_types=supported_eval_types
        self.output_path=output_path

    def tune_parameters(self,k,eval_type,eval_obj,fig_name='default.png',**params):
        if eval_type in self.supported_eval_types:
            self.eval_type=eval_type
        self.k=k['k']
        if self.eval_type=='ttsplit':
            indMin=0
            errMin=10000
            err=np.zeros(len(self.k))
            for i in self.k:
                self.model=self.model_class(i)
                x=self.train_test_split_eval(eval_obj,params['test_proportion'])['mean']
                xx=self.train_test_split_eval(eval_obj,params['test_proportion'])['std']
                err[i-1]=x+xx
                if (x+xx)<errMin:
                    errMin=x+xx
                    indMin=i
            plt.figure()
            plt.plot(self.k,err)
            plt.title('Mse al variare di k in ttsplit')
            plt.xlabel('k')
            plt.ylabel('Mse')
            plt.savefig(self.output_path+fig_name)
            return {'k': indMin}
        if self.eval_type=='kfold':
            indMin=0
            errMin=10000
            err=np.zeros(len(self.k))
            for i in self.k:
                self.model=self.model_class(i)
                x=self.kfold_cv_eval(eval_obj,params['K'])['mean']
                xx=self.kfold_cv_eval(eval_obj,params['K'])['std']
                err[i-1]=x+xx
                if (x+xx)<errMin:
                    errMin=x+xx
                    indMin=i
            plt.figure()
            plt.plot(self.k,err)
            plt.title('Mse al variare di k in kfold')
            plt.xlabel('k')
            plt.ylabel('Mse')
            plt.savefig(self.output_path+fig_name)
            return {'k': indMin}