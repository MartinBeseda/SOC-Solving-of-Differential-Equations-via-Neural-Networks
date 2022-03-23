import math
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np

Number_of_Neurons = 30
Intervalt = [0,2]
Intervalx = [0,2]

Jump = 0.1

#Making domain
X = []
d = Intervalx[0]
while d <= Intervalx[1]:
    X.append(d)
    d += Jump
T = []
d = Intervalt[0]
while d <= Intervalt[1]:
    T.append(d)
    d += Jump

#Weights and Biases
#operators
def function(x):
    return 1/(1+math.exp(-x))
def dfunction(x):
        return (function(x+0.001)-function(x))/0.001 #np.exp(-x) / (1 + np.exp(-x)) ** 2
def ddfunction(x):
    return  (dfunction(x+0.001)-dfunction(x))/0.001#(2*np.exp(-2*x))/(1+np.exp(-x))**3 - np.exp(-x)/(1+np.exp(-x))**2
#params = [W1 + W2 + B1], len(W1) = Number?of?Neurons
def value_of_N(x,t, params):
    #params = [wt,wx,w3,b,...]
    res = 0
    for i in range(0,4*Number_of_Neurons,4):
        res += params[i+2]*function(x*params[i+1]+t*params[i]+params[i+3])
    return res
def value_of_dNdx2(x,t,params):
    res = 0
    for i in range(Number_of_Neurons):
        res += params[i+1]*params[i+1]*params[i+2]*ddfunction(x*params[i+1]+t*params[i]+params[i+3])
    return res
def value_of_dNdt(x,t,params):
    res = 0
    for i in range(Number_of_Neurons):
        res += params[i]*params[i+2]*dfunction(x*params[i+1]+t*params[i]+params[i+3])
    return res

def value_of_dNdx(x,t,params):
    res = 0
    for i in range(Number_of_Neurons):
        res += params[i+1]*params[i+2]*dfunction(x*params[i+1]+t*params[i]+params[i+3])
    return res


def abserr(params):
    abserr = 0
    for x in X:
        if x == 0:
            for t in T:
                abserr += 100*(value_of_N(x,t,params) - math.sin(t))**2
        else:
            abserr += 100 * (value_of_N(x, 0, params) - math.exp(-x * 2 ** (-0.5)) * math.sin(2 ** (-0.5) * -x)) ** 2
            for t in T[1:]:
                err = 0
                err += (value_of_dNdt(x,t,params)-value_of_dNdx2(x,t,params))**2
                abserr += err
    print(abserr)
    return abserr

parameters = list(np.random.normal(size = 4*Number_of_Neurons))
ret = scipy.optimize.minimize(abserr, np.array(parameters), bounds = [(-30,30)]*len(parameters), options={'maxiter':200})
print(ret.success)
ret = list(ret.x)
print("Complete")
import matplotlib.patches as mpatches
x_graph = X
y1_graph = []
y2_graph = []
sinus = []
tl = []
for e in X:
    y1_graph.append(value_of_N(0,e, ret))
    y2_graph.append(value_of_N(e, 0, ret))
    sinus.append(math.sin(e))
    tl.append(math.exp(-e * 2 ** (-0.5)) * math.sin(2 ** (-0.5) * -e))

fig =  plt.figure(figsize= (9, 6))
ax_c = "green"
fig.suptitle('Nestálé proudění podzemní vody v 1D', fontsize=16, color = "red")
ax1 = fig.add_subplot(2,2,3)
ax2 = fig.add_subplot(2,2,4)
ax3 = fig.add_subplot(2,1,1)
ax1.plot(x_graph, y1_graph, "r-")
ax1.plot(x_graph, sinus, "g-")
plt.axis([Intervalt[0],Intervalt[1],min(y1_graph)-1,max(y1_graph)+1])
red_patch = mpatches.Patch(color='red', label='Analytické řešení')
blue_patch = mpatches.Patch(color='green', label='Neuronová síť')
ax1.legend(handles=[blue_patch, red_patch])
ax1.set_xlabel('Čas t', fontsize = 10, color = ax_c)
ax1.set_ylabel("Výstup funkce", fontsize = 10,  color = ax_c)
ax1.set_title('Okrajová podmínka pro x = 0', fontsize = 13, color = "red")
ax2.plot(x_graph, y2_graph, "g-")
ax2.plot(x_graph, tl, "r-")
plt.axis([Intervalt[0],Intervalt[1],min(y2_graph)-1,max(y2_graph)+1])
red_patch = mpatches.Patch(color='red', label='Analytické řešení')
blue_patch = mpatches.Patch(color='green', label='Neuronová síť')
ax2.legend(handles=[blue_patch, red_patch])
ax2.set_title('Okrajová podmínka pro t = 0', fontsize = 13, color = "red")
ax2.set_ylabel("Výstup funkce", fontsize = 10,  color = ax_c)
ax2.set_xlabel('Vzdálenost x', fontsize = 10,  color = ax_c)
plt.subplots_adjust(hspace = 0.3, wspace=0.3)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Make data.
N = []
Ndt = []
Ndx = []
for t in T:
    newN = []
    newNdt = []
    newNdx = []
    for x in X:
        newN.append(value_of_N(x,t,ret))
        newNdt.append(value_of_dNdt(x, t, ret))
        newNdx.append(value_of_dNdx(x, t, ret))
    N.append(newN)
    Ndx.append(newNdx)
    Ndt.append(newNdt)
N = np.array(N)
Ndt = np.array(Ndt)
Ndx = np.array(Ndx)
X, Y = np.meshgrid(X, T)

#countor
CS = ax3.contour(X, Y, N, 6)
ax3.clabel(CS, inline=True, fontsize=10)
ax3.set_title('Vrstevnicový graf', color = "red")
plt.axis([Intervalx[0],Intervalx[1],Intervalt[0],Intervalt[1]])
ax3.set_xlabel('Vzdálenost x', fontsize = 10,  color = ax_c)
ax3.set_ylabel('Čas t', fontsize = 10,  color = ax_c)
plt.show()
fig, ax = plt.subplots(figsize=(7, 5))
CS = ax.contour(X, Y, Ndt, 6)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Derivative podél t')
plt.xlabel('x ->')
plt.ylabel('t ->')
plt.show()
fig, ax = plt.subplots(figsize=(7, 5))
CS = ax.contour(X, Y, Ndx, 6)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Derivative podél x')
plt.xlabel('x ->')
plt.ylabel('t ->')
plt.show()


# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, N, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()