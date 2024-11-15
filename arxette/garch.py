"""implements zero mean GARCH(1,1) model with Gaussian noise"""

__all__ = ['GARCH']

from functools import cached_property
from typing import Literal
import numpy as np
# import jax.numpy as jnp
# from jax import jit, random
# from numba import njit
from scipy.optimize import minimize

# CONSTRAINT = {
#     "type": "ineq",
#     "fun": lambda x: 1-x[1]-x[2]
# }

def _get_vs(y, params, sig2_init):
    """
    not clear how arch package initialize the conditional variance at time 0
    seems very close to be the sample variance of y;
    it is provably true that vs computed below
    is insensitive to the initial value (coupling exponentially fast)
    """
    om,al,be = params
    vs = np.full(y.size, sig2_init)
    for i in range(1,y.size):
        vs[i] = om + al * y[i-1]**2 + be * vs[i-1]
    return vs

def _nllgauss(y,params,sig2_init):
    vs = _get_vs(y,params, sig2_init)
    nll =  np.log(vs) + y**2 / vs
    return nll.sum()

# @njit
def _make_fcst_vs(params,last_v, last_y, horizon):
    om,al,be = params
    fcst_vs = np.empty(horizon)
    fcst_vs[0] = om + al* last_y**2 + be* last_v
    for i in range(1,horizon):
        fcst_vs[i] = om + (al+be) * fcst_vs[i-1]
    return fcst_vs


# @njit
def _simulate(params, last_y, last_v, horizon, n_rep, seed, ws=None):
    om,al,be = params
    if ws is None:
        np.random.seed(seed)
        ws = np.random.randn(n_rep,horizon)
    y = np.empty((n_rep, horizon+1))
    v = np.empty((n_rep, horizon+1))
    y[:,0] = last_y
    v[:,0] = last_v
    for i in range(1,horizon+1):
        v[:,i] = om + al* y[:,i-1]**2 + be * v[:,i-1]
        y[:,i] = np.sqrt(v[:,i]) * ws[:,i-1]
    return y[:,1:]

class GARCH:
    """simple garch model"""
    def __init__(self):
        self._y = None
        self.params = None
        self._is_fit = False
        self._v_init = None

    def fit(self,y:np.ndarray):
        """fit y to garch"""
        self._y = y
        self._v_init = y.var() # != arch_model(y).fit().conditional_volatility[0]**2 but close 
        func = self.nll
        self.params = minimize(func, x0=(self._v_init * 0.4,0.3,0.3), 
         bounds=[(0,None), (0,None), (0,None)],
         constraints= {
            "type": "ineq",
            "fun": lambda x: 1-x[1]-x[2]
        }
         ).x
        self._is_fit = True
        return self

    def nll(self, params)-> float:
        """negative log likelihood of the series at the given params"""
        return _nllgauss(self._y, params, self._v_init)

    @cached_property
    def vs(self) -> np.ndarray:
        """property: estimated conditional variance, same shape as y"""
        assert self._is_fit
        return _get_vs(self._y, self.params, sig2_init = self._v_init)

    @cached_property
    def std_resids(self) -> np.ndarray:
        """property: estimated standardized residual"""
        assert self._is_fit
        return self._y / np.sqrt(self.vs)

    def forecast_vs(self, 
                    horizon:int
                    ) -> np.ndarray:
        """forecast conditional variance in the horizon"""
        assert self._is_fit
        return _make_fcst_vs(self.params, self.vs[-1], self._y[-1], horizon)

    def simulate(self, 
                 horizon:int, # path length
                 method:Literal["bootstrap","simulate"]="simulate",# "bootstrap" resamples from past std_resids; "simulate" simulates gaussian nosie
                 n_rep=1_000, # number of repetitions
                 seed=42) -> np.ndarray: 
        assert self._is_fit
        if method == "bootstrap":
            np.random.seed(seed)
            ws = np.random.choice(self.std_resids, size=(n_rep, horizon),replace=True)
        else: ws=None
        return _simulate(self.params,self._y[-1], self.vs[-1], horizon, n_rep, seed, ws)