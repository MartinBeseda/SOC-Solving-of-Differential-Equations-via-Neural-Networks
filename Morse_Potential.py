import scipy.optimize as so
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

Interval = (-1,1, 0.01)
Number_of_Neurons = 3
X = np.arange(Interval[0], Interval[1], Interval[2])
#np.savetxt('X_Morse.csv',np.asarray([X]), delimiter=';')

ground = np.loadtxt('Morse.csv', delimiter=';')
excite1 = np.loadtxt('Morse1.csv', delimiter=';')
excite2 = np.loadtxt('Morse2.csv', delimiter=';')

def activation(x):
    return 1/(1+np.exp(np.clip(-x, -100, 100)))

def NN(x:np.array,params):
    w1b = np.reshape(params[0:2 * Number_of_Neurons], (Number_of_Neurons, -1))
    w2 = params[2*Number_of_Neurons:3*Number_of_Neurons]
    a = np.reshape(np.append(x,np.ones(len(x))), (2,-1))
    z = np.matmul(w1b,a)
    return np.dot(w2,activation(z))*np.exp(-(params[-1]**2)*x**2)

def iNN(x:np.array,params):
    v1 = params[2*Number_of_Neurons:3*Number_of_Neurons]
    w1b = np.reshape(params[0:2 * Number_of_Neurons], (2,-1))
    z1 = w1b.transpose().dot(np.array([x[0],1]))
    z2 = w1b.transpose().dot(np.array([x[-1],1]))
    return np.dot(v1/w1b[0],1/(np.exp(z2)+1)+np.log(np.exp(z2)+1))-np.dot(v1/w1b[0],1/(np.exp(z1)+1)+np.log(np.exp(z1)+1))

def Hamiltonian(x:np.array, params):
    u = 119406
    a = 0.9374
    dx = 10e-5
    D = 0.0224
    T = -(NN(x + dx,params)-2*NN(x,params)+NN(x-dx,params)) / (2 * (dx ** 2) * u)
    V = D*(np.exp(-2*a*x)-2*np.exp(-a*x)+1)*NN(x,params)
    return T+V

def epsilon(params):
    a = scipy.integrate.trapz(NN(X,params)*Hamiltonian(X,params), x = X)
    b = scipy.integrate.trapz(NN(X,params)**2, x = X)
    return a/b

def err(params):
    #a = np.sum((Hamiltonian(X,params)-epsilon(params))**2)
    #b = scipy.integrate.trapz(NN(X,params)**2, x = X)
    return  epsilon(params)+1e8*(scipy.integrate.trapz(ground*NN(X,params),X))**2+1e7*(scipy.integrate.trapz(excite1*NN(X,params),X))**2+1e9*(scipy.integrate.trapz(excite2*NN(X,params),X))**2

eps = 10.0
#for _ in range(30):
init = np.random.normal(size=3 * Number_of_Neurons + 1)
print(scipy.integrate.trapz(NN(X,init)**2, x = X))
print(iNN(X,init))
ret = scipy.optimize.minimize(err,  x0 = init, method = "BFGS", tol = 1e-13, options={'maxiter': 10e3})
ret = list(ret.x)
#   if eps > epsilon(ret):
#       eps = epsilon(ret)
print("_______________________________")
print("Complete NN")
#print(ret.success)
#ret = list(ret.x)

print("Epsilon:", epsilon(ret))
print("Analytic:",0.286171979e-3)
#print("Norma:", NN(X,ret.x) / (np.sqrt(scipy.integrate.trapz(NN(X,ret)**2, x=X))))

#data = np.savetxt('Morse3.csv',np.asarray([NN(X,ret)/np.sqrt(scipy.integrate.trapz(NN(X,ret)**2, x = X))]), delimiter=';')

plt.plot(X,NN(X,ret)/np.sqrt(scipy.integrate.trapz(NN(X,ret)**2, x = X)))
plt.plot(X,NN(X,ret)**2/scipy.integrate.trapz(NN(X,ret)**2, x = X),"r-")
plt.legend(["VlnovÃ¡ funkce elektronu", "Hustota pravdÄ›podobnosti polohy"], loc ="upper right",prop={'size': 7})
plt.show()