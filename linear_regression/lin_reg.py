import numpy as np

def linear_regression(X, y):
    """
    Perform linear regression
    Find coefficients a such that y ~= aX 
    
    Parameters
    -----------

    X: np.ndarray, shape (n_samples, n_features)
        The input Data

    y: np.ndarray, shape (n_samples, )
        The target

    Returns
    -----------

    a: np.ndarray, containing the coeeficients
    
    """

    a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return a 
    