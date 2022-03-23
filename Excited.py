import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

E1 = np.loadtxt('Morse1.csv', delimiter=';')
E0 = np.loadtxt('Morse.csv', delimiter=';')
E2 = np.loadtxt('Morse2.csv', delimiter=';')
E3 = np.loadtxt('Morse3.csv', delimiter=';')
X = np.loadtxt("X_Morse.csv",delimiter = ";")

plt.plot(X,E0)
plt.plot(X,E1)
plt.plot(X,E2,"r-")
#plt.plot(X,E3)

plt.legend(["n = 0", "n = 1", "n = 2"], loc ="upper right",prop={'size': 7})


plt.show()
