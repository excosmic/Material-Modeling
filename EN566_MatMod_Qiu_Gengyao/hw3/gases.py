import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import sys
@njit
def gases_iter(iter_times=1000000):
    # init the state
    state = np.zeros((40, 60))
    for i in range(400): # This number is the # of gas atoms in each type.
        a_y, a_x = np.random.randint(0, 40), np.random.randint(0, 20)
        b_y, b_x = np.random.randint(0, 40), np.random.randint(40, 60)
        state[a_y, a_x], state[b_y, b_x] = 1, 2
    valid_update = 0
    while valid_update<iter_times:
        y_direct, x_direct = np.random.randint(-1, 2), np.random.randint(-1, 2)
        y, x = np.random.randint(0, 40), np.random.randint(0, 60)
        if(y+y_direct in range(0, 40)) and (x+x_direct in range(0, 60)):
            if state[(y+y_direct)%40, (x+x_direct)%60]==0:
                state[(y+y_direct)%40, (x+x_direct)%60] = state[y, x]
                state[y, x]=0
                valid_update+=1
    return state

def density(attempt=1, iter_times=10000):
    d_a, d_b = np.zeros(60), np.zeros(60)
    for i in range(attempt):
        print(i)
        state = gases_iter(iter_times=iter_times)
        state_a, state_b = (state==1)*state, (state==2)*state
        d_a += state_a.sum(0)
        d_b += state_b.sum(0)/2
    return d_a/attempt, d_b/attempt

def get_part():
    arg = sys.argv[1][7:]
    arg = int(arg)
    return arg

if __name__=='__main__':
    match get_part():
        case 1:
            print('please see the answer on the ')
        case 2:
            d_a, d_b = density(attempt=100, iter_times=500000)
            plt.plot(d_a, label='gas A'); plt.plot(d_b, label='gas B')
            plt.legend();plt.title(f'Density of gases in time={500000}')
            plt.xlabel('x');plt.ylabel('n')
            plt.show()
            for i in [0, 1000, 100000, 1000000]:
                stat = gases_iter(iter_times=i)
                plt.matshow(stat)
                plt.title(f'State of Gases in time={i}')
                plt.show()
            pass
    pass