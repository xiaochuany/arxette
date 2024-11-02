import pytest
import numpy as np

from arx import ARX

# simulate arx process
x = np.random.rand(2000, 5)*2 + 4
w = np.array([1,2,3,4,5])
b = 0
al = -0.1
v=1

y=np.zeros(2000)
wn = np.random.randn(2000)*np.sqrt(v)

for i in range(1,2000):
    y[i] = x[i].dot(w) + b + al*y[i-1] + wn[i]

def test_close():
    mod = ARX()
    pnll = mod.fit(y,x).params[:5]
    pols = mod.fit_ols(y,x).params_ols[:5]
    assert np.isclose(pnll,pols, atol=0.2).all()


if __name__ == "__main__":
    pytest.main()