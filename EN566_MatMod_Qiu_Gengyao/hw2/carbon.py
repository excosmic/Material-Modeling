import numpy as np
import sys
import matplotlib.pyplot as plt
NA = 6.022e23
N0 = NA*10e-9/14
half=5700
tau = half/np.log(2)
years = 20000
print("%e"%N0)

def decay(step_width:int, N0:float=N0, tau:float=tau, years:int=years)->list[float]:
    N = [N0]
    for i in range(years//step_width):
        N += [(-1/tau)*N[-1]*step_width+N[-1]]
    return np.array(N)
if __name__=='__main__':
    if sys.argv[1][:7]!='--plot=': raise Exception('Expect --plot=')
    anay = lambda y:1/tau*N0*np.exp(-1/tau*(y))
    ay = anay(ax:=np.arange(0, 20001, 1))
    plt.plot(ax, ay, label='Analytical R(t)', linestyle='-', c='yellow')
    for width in sys.argv[1][7:].split(','):
        if width!='':
            R = decay(int(width))/tau
            # Calculate the deviation
            real:float = 2*half/int(width); Pi:int = int(real); Pi1:int = Pi+1; 
            R_numerical = (real-Pi)/(Pi1-Pi)*(R[Pi1]-R[Pi])+R[Pi]
            R_analysis = anay(half*2)
            print(f'The deviation in width={width} is {(R_numerical-R_analysis)/R_analysis*100}%')
            plt.plot(np.arange(0, 20001, int(width)), R, label='Euler dt='+str(width), linestyle='--', lw=0.7)
    plt.xlabel('Year'); plt.ylabel('R(t)'); plt.title('Decay of Carbon14'); plt.legend()
    plt.show()
