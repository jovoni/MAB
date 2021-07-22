import numpy as np
from scipy.stats import norm

def UCB(mu,s,k=1):
    return mu + k*s

def GP_UCB(mu, s, xmin, xmax, t, delta = 0.5):
    D = abs(xmax - xmin)
    beta = 2 * np.log(D * t**2 * np.pi**2 / (6*delta))
    return mu + beta*s

def PI(y, mu, s, eps=0.1):
    best_y = np.max(y)
    num = mu - best_y - eps
    return norm.cdf(num / s)

def EI(y, mu, s, eps=0.1):
    best_y = np.max(y)
    t1 = (mu - best_y - eps)
    t2 = norm.cdf(t1 / s)
    t3 = norm.pdf(t1 / s)
    return t1 * t2 + s * t3

def TS(X, y, mu, Xtest, kernel, noise):
    n = len(Xtest)
    N = len(y)
    
    K = kernel(X, X)
    L = np.linalg.cholesky(K + noise*np.eye(N))
    Lk = np.linalg.solve(L, kernel(X, Xtest))
    K_ = kernel(Xtest, Xtest)
    # draw a sample from the posterior
    L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
    return mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,1)))