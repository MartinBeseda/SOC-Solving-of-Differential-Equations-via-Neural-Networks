import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.constants as c
import scipy.misc
import warnings

warnings.filterwarnings('error')

Number_of_Neurons = 6

L=1
X = np.linspace(0, L, 50)

me = 5.4858*1e-4

def V(x: np.array):
    return 0


def Tpsi(x: np.array, params):
    return -c.value('reduced Planck constant') / (2 * c.electron_mass) * scipy.misc.derivative(lambda p: psi(p, params),
                                                                                               x0=x, dx=1e-5, n=2,
                                                                                               order=3)

def epsilon(x, params):
    i = 1 / (scipy.integrate.trapz(psi(x, params) ** 2, x))
    c = scipy.integrate.trapz(psi(x, params) * (Tpsi(x, params) + V(x) * psi(x, params)), x)

    return (c) * i


# Neural Network
def activation(x):
    return 1 / (1 + np.exp(np.clip(-x, -1e2, 1e2)))


def N(x: np.array, params):
    n_net_params = len(params) - 1
    n_type_params = n_net_params // 3
    w1 = params[0: n_type_params]
    b = params[n_type_params: 2 * n_type_params]
    w2 = params[2 * n_type_params: 3 * n_type_params]
    return activation(np.tile(x, (n_type_params, 1)).T * w1 + b) @ w2


def psi(x: np.array, params):
    # return x * R(x, params)
    return x * (L-x) * N(x, params)


def err(params):
    # return epsilon(X, params)
    return np.sum((Tpsi(X, params)
                   + V(X) * psi(X, params) - epsilon(X, params) * psi(X, params)) ** 2) / (
               scipy.integrate.trapz(psi(X, params) ** 2, X))


if __name__ == '__main__':
    print(f'Number of neurons: {Number_of_Neurons}')

    init = np.random.normal(size=3 * Number_of_Neurons)
    print(f'Init params: {init}')
    print("Error:", err(init))
    print("Epsilon:", epsilon(X, init))

    print('\nLocal optimization starting...')
    ret = scipy.optimize.minimize(err, tol=1e-6, x0=init, method="L-BFGS-B", options={'maxiter': 1000},
                                  bounds=[(10, 20)]*Number_of_Neurons + [(2, 3)]*Number_of_Neurons + [(0.5, 1.5)]*Number_of_Neurons)
    new_params = ret.x

    print("_______________________________")
    print("Complete NN")
    print(ret.success)
    print("Error:", err(new_params))
    print("Epsilon:", epsilon(X, new_params))
    print(f"Params: {new_params}")

    res = psi(X, new_params)

    plt.plot(X, np.sqrt(2/L)*np.sin(((np.pi*X)/L)), label='analytical')
    plt.plot(X, psi(X, init), label='psiOld')
    plt.plot(X, psi(X, new_params), label='psi')
    plt.legend()
    plt.show()
