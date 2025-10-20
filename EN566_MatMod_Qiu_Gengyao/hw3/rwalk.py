import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import sys
@njit
def one_random_walk(n, xs=1, ys=1):
    ans = [[0.0, 0.0]]
    for i in range(n):
        dx, dy = 0, 0
        while dx==0 and dy==0: dx, dy = np.random.randint(-1, 2), np.random.randint(-1, 2)
        ans += [[ans[-1][0]+dx, ans[-1][1]+dy]]
    return np.array(ans)
def first_task():
    ans_1 = []
    ans_2 = []
    for i in range(10000):
        rw = one_random_walk(n=50)
        ans_1 += [rw]
        ans_2 += [rw**2]
    ans_1, ans_2 = np.array(ans_1).mean(axis=0), np.array(ans_2).mean(axis=0)
    print(ans_1.shape, ans_2.shape)
    plt.plot(ans_1[:, 0], label="<x_n>")
    plt.plot(ans_2[:, 0], label='<x_n^2>')
    plt.title('Average of Random Walk');plt.legend()
    plt.xlabel('step');plt.ylabel('average')
    plt.show()
    return

def second_task():
    ans = []
    for i in range(10000):
        ans += [one_random_walk(n=49)]
    ans = np.array(ans)
    print(ans.shape)
    ans = (ans[:, :, 0]**2 + ans[:, :, 1]**2)**1
    ans = ans.mean(axis=0)
    plt.plot(ans)
    plt.title('Mean Square Distace from Start Point'); plt.xlabel('time'); plt.ylabel('<r^2>')
    plt.show()
# first_task()

# second_task()
def get_part():
    arg = sys.argv[1][7:]
    arg = int(arg)
    return arg

if __name__=='__main__':
    match get_part():
        case 1:
            first_task()
        case 2:
            second_task() 
    pass