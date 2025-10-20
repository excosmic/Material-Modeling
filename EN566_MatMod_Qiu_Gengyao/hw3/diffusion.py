import numpy as np
import matplotlib.pyplot as plt
import sys

def diffusion_equation(D=2, delta_t=0.001, delta_x=0.1, shape=1000):
    rho = np.zeros(shape)    
    rho[shape//2-1:shape//2+1] = 1
    rhos, peaks, times = [], [], []
    for i in range(50000):
        rho = rho / (np.sum(rho) * delta_x) # normalization
        rho = rho + D*delta_t/(delta_x**2)*(np.roll(rho, 1, 0)+np.roll(rho, -1, 0)-2*rho)
        if i!=0 and i%10000==0: rhos += [rho]; peaks += [rho.max()]; times += [delta_t*i]
    sigmas = 1/((2*np.pi)**0.5)/np.array(peaks)
    return rhos, np.array(times), sigmas

def get_part():
    arg = sys.argv[1][7:]
    arg = int(arg)
    return arg

if __name__ == '__main__':
    match get_part():
        case 1:
            print('please see the answer in pdf')
        case 2:
            rhos, times, sigmas = diffusion_equation()
            print(f'squr(Dt)={(4*times)**0.5}, sigma={sigmas}')
            for i in range(len(rhos)):
                plt.plot(rhos[i], label=f'distribution at t={times[i]}')
            plt.legend()
            plt.title('Distribution in Different Time')
            plt.show()