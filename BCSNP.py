import numpy as np

def CS_BCSNP(y:np.ndarray, A:np.ndarray, maxiter:int)->np.ndarray:
    '''
    Zhou, Zhou, Kaihui Liu, and Jun Fang. "Bayesian compressive sensing using normal product priors." IEEE Signal Processing Letters 22.5 (2014): 583-587.

    Based on matlab code from: https://github.com/KaihuiLau/BCSNP

    Input:
        y: measurement vector
        A: measurement matrix
        maxiter: maximum iterations

    Output:
        est_x: reconstructed vector
    '''
    M,N = A.shape
    kappa = 0.1 * np.ones(N)
    gamma = 0.1 * np.ones(N)

    mean_a = 1 * np.ones(N)
    mean_b = 1 * np.ones(N)
    converged = False
    iter = 0
    x_new = mean_a * mean_b
    Num = N

    while not converged:
        x_old = x_new

        #update b
        var_b = (np.eye(Num) - np.diag(gamma) @ np.linalg.pinv(A@np.diag(mean_a)@np.diag(gamma)) @ A @ np.diag(mean_a)) @ np.diag(gamma**2)

        mean_b = np.diag(gamma) @ np.linalg.pinv(A @ np.diag(mean_a) @ np.diag(gamma)) @ y
        mean_b = mean_b.reshape(-1)

        #update a
        _b = A@np.diag(mean_b)
        var_a = (np.eye(Num) - np.diag(kappa) @ np.linalg.pinv(A@np.diag(mean_b)@np.diag(kappa)) @ A @ np.diag(mean_b)) @ np.diag(kappa**2)

        mean_a = np.diag(kappa) @ np.linalg.pinv(A @ np.diag(mean_b) @ np.diag(kappa)) @ y
        mean_a = mean_a.reshape(-1)

        #update kappa
        kappa = np.sqrt(mean_a**2 + np.diag(var_a))

        #update gamma
        gamma = np.sqrt(mean_b**2 + np.diag(var_b))

        x_new = mean_a * mean_b

        if np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old) < 1e-9 or iter >= maxiter:
            converged = True

        iter += 1

    est_x = x_new
    est_s = (np.abs(x_new)>1e-3)
    return est_x, est_s