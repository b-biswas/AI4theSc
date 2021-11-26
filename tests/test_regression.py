import numpy as np
from linear_regression import linear_regression # for this to work you need to have run the pip install command

def test_dimension():
    X = np.random.randn(3,2)
    y = np.random.randn(3)
    coefs = linear_regression(X,y)
    assert len(coefs)==2