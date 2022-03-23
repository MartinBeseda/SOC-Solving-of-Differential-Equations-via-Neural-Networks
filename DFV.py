import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

inputs = (0.0, 4.0)

df = [(e, 0) for e in np.arange(inputs[0], inputs[1]+0.1, 0.1)]


def fx(inp, parameters):
    sum_total = 0
    for i in list(range(0, len(parameters), 3)):
        # w2*s(w1*x + b)
        sum_total += parameters[i+2]*sigmoid_f(parameters[i+1]*inp+parameters[i])
    return sum_total


def dfx(inp, parameters):
    sum_total = 0
    for i in list(range(0, len(parameters), 3)):
        # w2*w1*s'(w1*x + b)
        sum_total += parameters[i+2]*parameters[i+1]*sigmoid_df(parameters[i+1]*inp+parameters[i])
    return sum_total


def ddfx(inp, parameters):
    sum_total = 0
    for i in list(range(0, len(parameters), 3)):
        # w2*w1*w1*s''(w1*x + b)
        sum_total += parameters[i+2]*parameters[i+1]*parameters[i+1]*sigmoid_ddf(parameters[i+1]*inp+parameters[i])
    return sum_total


def sigmoid_f(inp):
    return 1/(1+np.exp(-inp))


def sigmoid_df(inp):
    return np.exp(-inp)/(1+np.exp(-inp))**2


def sigmoid_ddf(inp):
    return (2*np.exp(-2*inp))/(1+np.exp(-inp))**3 - np.exp(-inp)/(1+np.exp(-inp))**2


def error(parameters):
    err_sum = 0
    for e in df:
        if e[0] == 0:
            err_sum += 300*(fx(e[0], parameters)-1)**2 + 300*(dfx(e[0], parameters)-1)**2

        err_sum += (ddfx(e[0], parameters) + 4*(dfx(e[0], parameters)) + 4*(fx(e[0], parameters)))**2
    return err_sum


if __name__ == '__main__':
    number_of_neurons = 10
    params = list(np.random.normal(size=number_of_neurons*3))
    x0 = np.array(params)
    res = minimize(error, x0, bounds=[(-25, 25)]*len(params))
    print(res)
    res = list(res.x)
    print('\nDone \n')

    for e in np.arange(inputs[0], inputs[1]+0.1, 0.1):
        print(round(e, 1), fx(round(e, 1), res))

    x_graph = np.arange(inputs[0], inputs[1]+0.1, 0.1)
    plt.plot(x_graph, fx(x_graph, res), '*', label='neuronová síť')
    plt.plot(x_graph, np.exp(-x_graph*2)*(1+3*x_graph), 'r', label='analytické řešení')
    # plt.plot(x_graph, dfx(x_graph, res), 'r')
    # plt.plot(x_graph, ddfx(x_graph, res), "g-")
    plt.xlabel('Čas', fontsize=9, fontweight='bold')
    plt.ylabel('Amplituda', fontsize=9, fontweight='bold')
    plt.title('Graf průběhu lineárního tlumeného oscilátoru', fontsize=14)
    plt.legend()
    plt.show()
    plt.plot(x_graph, dfx(x_graph, res), 'r*', label='neuronová síť')
    plt.plot(x_graph, np.exp(-x_graph*2)*(1-6*x_graph), 'g', label='analytické řešení')
    plt.xlabel('Čas', fontsize=9, fontweight='bold')
    plt.ylabel('Amplituda', fontsize=9, fontweight='bold')
    plt.legend()
    plt.show()
    plt.plot(x_graph, ddfx(x_graph, res), 'g*', label='neuronová síť')
    plt.plot(x_graph, np.exp(-x_graph*2) * (-8+12*x_graph), 'b', label='analytické řešení')
    plt.xlabel('Čas', fontsize=9, fontweight='bold')
    plt.ylabel('Amplituda', fontsize=9, fontweight='bold')
    plt.legend()
    plt.show()
