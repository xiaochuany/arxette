# arxette

autoregressive model with exogeneous variables

## install

```sh
$ pip install arxette
```

## how to use

```python
import numpy as np
from arxette import ARX, GARCH
```

```python
y = np.random.randn(20)
x = np.random.randn(20,5)
mod = ARX()
mod.fit(y,x) 
```

The `fit` method minimize the nll. Alternative one could use the OLS estimator with 

```python
mod.fit_ols(y,x)
```

Once the model is fit, one can inspect the fitted parameters 

```python
mod.params # nll 
mod.params_ols # ols
```

## test

To run the tests, install `pytest` and run

```sh
pytest
```

which tests that the params fit by both methods are pretty close. 



