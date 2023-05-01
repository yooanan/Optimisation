import numpy as np
from numpy import genfromtxt
from scipy.stats import ncx2
from scipy.optimize import minimize


def cirpdf(y, t, a, b, sigma):
    d = np.exp(-a * (t[1:] - t[:-1]))
    c = 2 * a / (sigma ** 2 * (1 - d))
    q = 2 * a * b / (sigma ** 2) - 1
    z = 2 * c * y[1:]  # <- transformed variable
    _lambda = 2 * c * y[:-1] * d  # <- non-centrality
    df = 2 * q + 2  # <- degrees of freedom
    if not (2 * a * b > sigma ** 2):  # <- check feller condition
        res = np.full(len(t) - 1, 1e-100)  # <- if fails return negligible probability density
    else:
        res = 2 * c * ncx2.pdf(z, df, _lambda)  # <- else return real probability density.
    return res



my_data = genfromtxt('Optimisation/CIRDataSet-2.csv', delimiter=',')
cirdata = np.delete(my_data,(0),axis=0)
t = cirdata[:,0]
xir = cirdata[:,1]
n = len(xir)
#initial guess must be 2ab ≥ σ^2 must hold
initial_guess=[1, 1, 0.5]

def summ(params):
    a,b,c = params
    x = np.zeros(n)
    for i in range(n-2):
        yi = xir[i:i+2]
        ti = t[i:i+2]
        x[i] = -np.log(cirpdf(yi,ti,a,b,c))
    return sum(x)


a1 = np.zeros(n-1)
a2 = np.zeros(n-1)
a3 = np.zeros(n-1)
a4 = np.zeros(n-1)

for i in range(n-1):
    a1[i] = xir[i+1]
    a2[i] = 1/xir[i]
    a3[i] = xir[i]
    a4[i] = xir[i+1]/xir[i]

a = (n**2-2*n + 1 + sum(a1) * sum(a2) - sum(a3) * sum(a2) - (n-1) * sum(a4))/((n**2-2*n+1-sum(a3)*sum(a2))*0.5)
b = ((n-1)*sum(a1) - sum(a4)*sum(a3))/(n**2-2*n + 1 + sum(a1) * sum(a2) - sum(a3) * sum(a2) - (n-1) * sum(a4))

x = np.zeros(n-1)
for i in range(n-1):
    x[i] = xir[i+1] - xir[i]

dx = x/xir[0:999]**0.5
reg = np.array([0.5/xir[0:999]**0.5, 0.5*xir[0:999]**0.5])
drift = np.array([a*b, -a])
res = np.array([reg[0]*drift[0] + reg[1]*drift[1] - dx])
sig = np.sqrt(np.var(res)/0.5)
sig = 0.0658
inigu = [a,b,sig]
inigu = [1,1,0.5]
re = minimize(summ,inigu)

sol = [5.54598146, 0.04957456, 0.15250057]

summ(sol)
