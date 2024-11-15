"""implements ARX model with Gaussian noise"""

__all__ = ['ARX']

import numpy as np 
from scipy.optimize import minimize 

def _nllgauss(y,x,params):
    """
    x:(m,d)
    y:(m,)
    params: (d+3,)
    """
    _,d = x.shape
    w= params[:d]
    b,v,al = params[d:]
    nll = np.log(v) + (y-x@w-b-al*np.roll(y,1))**2/v
    return nll[1:].sum()

class ARX:
    def __init__(self):
        self.params=None
        self._is_fit=False
        self._y = None
        
    def fit(self,y,x):
        self._y = y
        self._x = x
        m,d= x.shape
        func = self.nll
        self.params = minimize(
            func, 
            x0= np.concatenate([np.random.randn(d), np.array([0, 1,0])] ),
            bounds=[(None,None) for _ in range(d+1)] + [(1e-6,None), (-1+1e-6,1-1e-6)],
        ).x
        self._is_fit=True
        return self
        
    def fit_ols(self,y,x):
        self._y = y
        self._x = x
        m,d = x.shape
        assert y.size==m
        X = np.concatenate([x, np.roll(y,1)[:,None]],axis=-1)
        self.params_ols = np.linalg.pinv(X.T @ X) @ X.T @ y
        self._is_fit=True
        return self
        
    def nll(self,params):
        return _nllgauss(self._y,self._x,params)

    def forecast(self, x):
        m,d = x.shape
        w= self.params[:d]
        b,v,al = self.params[d:]
        fcs = np.zeros(m+1)
        y_last = self._y[:-1]
        