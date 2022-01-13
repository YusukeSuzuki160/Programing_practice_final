import numpy as np
import math
    
def mean_vec(x):
    n1 = x.shape[0]
    n2 = x.shape[1]
    av = np.full((1, n2), 0.0)
    for i in range(n1):
        av += x[i]
    return av / n1
    
def covarianceMatrix(x):
    n1 = x.shape[0]
    n2 = x.shape[1]
    y = np.full((n2, n2), 0.0)
    y = np.asmatrix(y)
    for i in range(n1):
        x2 = np.asmatrix(x[i])
        y += x2.T @ x2
    return np.asmatrix(y / n1)

def diagonalization(a):
    w, v = np.linalg.eig(a)
    n = a.shape[1]
    b = np.full((n, n), 0.0)
    for i in range(n):
        b[i][i] = w[i]
    return b, v

def generate_V(b, v):
    n = b.shape[0]
    c = np.full((n, n), 0.0)
    for i in range(n):
        r = math.sqrt(b[i][i])
        c[i][i] = 1 / r
    return v @ c @ v.T

def whitening(x, d):
    return d @ x.T

def solve(w, z):
    z2 = np.full((z.shape[0], z.shape[1]), 0.0)
    z2 = np.asmatrix(z2)
    for i in range(z.shape[0]):
        y = np.dot(w, z[i])
        z2[i] = y ** 3 * np.asmatrix(z[i])
    w1 = mean_vec(z2) - 3 * w
    w2 = w1 / np.linalg.norm(w1, ord = 2)
    check1 = w2 - w
    check2 = w2 + w
    if np.linalg.norm(check1, ord = 2) < 0.00001:
        return w2 
    elif np.linalg.norm(check2, ord = 2) < 0.00001:
        return (-1) * w2
    else:
        return solve(w2, z)        
        
def answer(w, x, n):
    m = x.shape[0]
    y = np.full((m, n), 0.0)
    y = np.asmatrix(y)
    wout = np.full((n, n), 0.0)
    sigma = covarianceMatrix(x)
    d, v = diagonalization(sigma)
    d = generate_V(d, v)
    n1 = x.shape[0]
    n2 = x.shape[1]
    z = np.full((n1, n2), 0.0)
    for i in range(n1):
        z[i] = whitening(x[i], d).T
    for i in range(n):
        wout[i] = solve(w[i], z)
    for i in range(m):
        y[i] = (np.asmatrix(wout) @ np.asmatrix(z[i]).T).T
    return y
        