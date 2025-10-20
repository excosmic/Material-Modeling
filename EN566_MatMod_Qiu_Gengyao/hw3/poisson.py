import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import sys

def jacobi_core(h=0.06, a=0.6, shape=(300, 300), tolerance=0.0001):
    xy2idx = lambda y, x: (shape[0]//2+int(y/h), shape[1]//2+int(x/h))
    v = np.zeros(shape)
    rho = np.zeros(shape)
    # set point charge
    rho[xy2idx(0, -a/2)] = -1
    rho[xy2idx(0, +a/2)] = +1
    err = tolerance+1; last = v.copy()+3
    n_iter = 0
    while err>tolerance:
        v = (np.roll(v, 1, 0)+np.roll(v, 1, 1)+np.roll(v, -1, 0)+np.roll(v, -1, 1)+rho)*0.25
        v[0, :] = v[:, 0] = v[:, -1] = v[-1, :] = 0 # set boundary
        err = np.abs((v-last)/(v-1e-10)).max()
        if n_iter%1000==0:print(err)
        last = v.copy()
        n_iter+=1
    #print(v)
    #plt.matshow(v[400:600, 400:600])
    #plt.contour(v[400:600, 400:600], 80, colors='k', linewidths=0.1)
    #plt.show()
    return v, n_iter

@njit
def sor_core(h=0.06, a=0.6, shape=(300, 300), tolerance=0.0001, omega=0.9):
    xy2idx = lambda y, x: (shape[0]//2+int(y/h), shape[1]//2+int(x/h))
    sy, sx = shape[0], shape[1]
    v = np.zeros(shape)
    rho = np.zeros(shape)
    # set point charge
    rho[xy2idx(0, -a/2)] = -1
    rho[xy2idx(0, +a/2)] = +1
    err = tolerance+1; last = v.copy()+3
    n_iter = 0
    while err>tolerance:
        v[0, :] = v[:, 0] = v[:, -1] = v[-1, :] = 0 # set boundary   
        for i in range(1, shape[0]-1):
            for j in range(1, shape[1]-1):
                v[i, j] = (1-omega)*v[i, j] + 0.25*omega*(v[i+1, j]+v[i-1, j]+v[i, j+1]+v[i, j-1]+rho[i, j])
        err = np.abs((v-last)/(v+1e-10)).max()
        last = v.copy()
        n_iter += 1
        #print(n_iter, err)
    return v, n_iter

def iter_of_error():
    n_iter_jacobi=[]
    n_iter_sor = []
    x = []
    for i in range(3, 13):
        _, n_jacobi = jacobi_core(tolerance=10**(-i),shape=(100, 100))
        _, n_sor = sor_core(tolerance=10**(-i), shape=(100, 100))
        n_iter_jacobi += [n_jacobi]; n_iter_sor += [n_sor]
        x += [i]
        print(i)
    #print(n_iter)
    plt.plot(x, n_iter_jacobi, label='Jacobi')
    plt.plot(x, n_iter_sor, label='SOR')
    plt.xlabel('log(Accuracy)'); plt.ylabel('number of iteration'); plt.title('Relationship Between Accuracy and Number of Iteration')
    #plt.loglog(basex=10, basey=10)
    #plt.xscale('log')
    plt.legend()
    plt.show()
    pass

def iter_n():
    jacobi_n, jacobi_iter = [], []
    sor_n, sor_iter = [], []
    sor_core()
    for i in range(200, 300, 10):
        _, j_iter = jacobi_core(shape=(i, i), tolerance=0.001)
        _, s_iter = sor_core(shape=(i, i), tolerance=0.001, omega=1.7)
        sor_n += [i]; jacobi_n += [i**2]
        sor_iter += [s_iter]; jacobi_iter += [j_iter]
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), constrained_layout=True)
    axes[0].plot(sor_n, sor_iter, 'o-'); axes[1].plot(jacobi_n, jacobi_iter, 'o-')
    axes[0].set_title('In SOR Method'); axes[0].set_xlabel('n'); axes[0].set_ylabel('number of iteration')
    axes[1].set_title('In Jacobi Method'); axes[1].set_xlabel('n^2'); axes[0].set_ylabel('number of iteration')
    #plt.plot(jacobi_n, jacobi_iter, 'o-')
    plt.show()
        
def drawmat(v):
    plt.matshow(v)
    plt.contour(v, 80, colors='k', linewidths=0.1)
    plt.show()

def get_part():
    arg = sys.argv[1][7:]
    arg = int(arg)
    return arg

if __name__=='__main__':
    #calculate_core()
    #iter_of_error()
    #v, _ = jacobi_core(shape=(200, 200), tolerance=0.0001)
    #drawmat(v)
    match get_part():
        case 1:
            v, n_iter = jacobi_core(a=1.8, shape=(100, 100), tolerance=0.0001)
            drawmat(v)
            directory = v[v.shape[0]//2, v.shape[1]//2:]
            print(directory.shape)
            x = np.linspace(0.0001, 100, len(directory))**-2
            plt.plot(x[40:], directory[40:], 'o-')
            plt.xlabel('r^-2'); plt.ylabel('dipole potential'); plt.title('Large-Distance Behavior of Dipole Potential')
            #plt.legend()
            plt.show()
        case 2:
            iter_of_error()
            pass
        case 3:
            #sor_core(omega=1.5)
            iter_n()
            pass
        case 4:
            _, n = sor_core(tolerance=0.00001)
            print(n)