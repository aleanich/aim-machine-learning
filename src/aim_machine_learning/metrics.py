
import numpy as np

class Evaluator:
    def __init__(self,supported_metrics):
        self.supported_metrics=supported_metrics
        

    def set_metric(self,new_metric):
        if (new_metric in self.supported_metrics):
            self.new_metric=new_metric
        else:
            raise NameError
        return self
    
    def current_metric(self):
        return self.new_metric
    
    def __repr__(self):
        return 'Current metric is {}'.format(self.new_metric)

    def __call__(self,y_true,y_pred):
        if self.new_metric=='mse':
          mse=np.mean((y_true-y_pred)**2)
          std=np.std((y_true-y_pred)**2)
          return {'mean': mse, 'std': std}
    
        if self.new_metric=='mae':
          diff=abs(y_true-y_pred)
          mae=np.mean(diff)
          std=np.std(diff)
          return {'mean': mae, 'std': std} 
        
        if self.new_metric=='corr':
            cov=np.corrcoef(y_true,y_pred)
            return {'corr': cov[0][1]/np.sqrt(cov[0][0]*cov[1][1])}
    
    
        
