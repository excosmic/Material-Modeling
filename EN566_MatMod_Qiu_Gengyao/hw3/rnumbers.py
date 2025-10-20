import numpy as np
import matplotlib.pyplot as plt
import sys
def first_task(bins=10, n=1000000):
    r = np.random.rand(n)
    plt.hist(r,bins=bins)
    plt.title(f'Distribution of Random Number with {bins} subdivisions')
    plt.show()

def second_task(bins=100, n=1000000):
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    norm = ((-2*np.log(u1))**0.5)*np.cos(2*np.pi*u2)
    x = np.linspace(-4, 4, 100)
    y = 1/((2*np.pi)**0.5)*np.exp(-(x**2)/2)
    plt.hist(norm,bins=bins, density=True)
    plt.title('Gaussian Distribution')
    plt.plot(x, y)
    plt.show()

#second_task()
#first_task(50)

def get_part():
    arg = sys.argv[1][7:]
    arg = int(arg)
    return arg

if __name__=='__main__':
    match get_part():
        case 1:
            first_task(10)
            first_task(20)
            first_task(50)
            first_task(100)
        case 2:
            second_task()
        