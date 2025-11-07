import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from sys import argv
def init(n:int=50):
    return np.random.randint(2, size=(n,n))*2-1

def get_energy(state, x, y, j:float=1.5):
    h = (np.roll(state, -1, 0)*state\
        + np.roll(state, -1, 1)*state).sum()*(-j)
    return h

@nb.njit
def get_total_energy(state, j=1.5):
    energy = 0.0
    ylen, xlen = state.shape[0], state.shape[1]
    for y in range(ylen):
        for x in range(xlen):
            energy += -j*state[y, x]*state[y, (x+1)%xlen]
            energy += -j*state[y, x]*state[(y+1)%ylen, x]
    return energy

@nb.njit
def energy_change(state, x, y, j:float=1.5):
    neighbor_add = lambda state, x, y:\
        state[(y-1)%state.shape[0], x%state.shape[1]]\
        + state[y%state.shape[0], (x-1)%state.shape[1]]\
        + state[(y+1)%state.shape[0], x%state.shape[1]]\
        + state[y%state.shape[0], (x+1)%state.shape[1]]
    x, y = x%state.shape[1], y%state.shape[0]
    
    return 2*j*state[y, x]*neighbor_add(state, x, y)
@nb.njit
def get_m(state):
    return np.abs(state.sum())/state.shape[0]/state.shape[1]
@nb.njit
def step(state, st:int=10, t=3, kb=1, j=1.5, average_sample_rate=0.5):
    N = state.shape[0]*state.shape[1]
    m, mi=0, 0
    current_E = get_total_energy(state, j)
    acc_E, acc_E2 = 0.0, 0.0
    for i in range(st):
        for j in range(state.shape[0]**2):
            y, x = np.random.randint(0, state.shape[0]), np.random.randint(0, state.shape[1])
            dE = energy_change(state, x, y)
            accept = False
            if dE<0:
                state[y, x] = -state[y, x]
                accept=True
            else:
                p_accept = np.exp((-dE)/(kb*t))
                if np.random.rand()<p_accept:
                    state[y, x] = -state[y, x]
                    accept=True
            if accept:
                current_E += dE
        if i>(1-average_sample_rate)*st:
            m += get_m(state)
            acc_E += current_E
            acc_E2 += current_E**2
            mi+=1
    return state, m/mi, m/mi/N, acc_E/mi/N, acc_E2/mi/N/N

def part2(kb=1):
    n_len = [5, 10, 20, 30, 40, 50]
    t_range = np.linspace(2.5, 4.5, 60)
    c_max_per_N_list = []
    n_list = []
    plot_data = {}
    for  n in n_len:
        print(n)
        N=n**2
        c_per_N_vs_T = []
        for T in t_range:
            state = init(n)
            state, _, m_avg, e_avg_ps, e2_avg_ps2 = step(
                state, st=25000, t=T, kb=1
            )
            c_per_N_vs_T += [(N*(e2_avg_ps2-(e_avg_ps**2)))/(kb*T*T)]
        c_max_per_N_list += [np.max(c_per_N_vs_T)]
        n_list += [n]
        plot_data[n] = c_per_N_vs_T
    #
    plt.figure()
    plt.plot(np.log(n_list), c_max_per_N_list, label='Simulation Data')
    plt.xlabel('$\log(n)$')
    plt.ylabel('Peak Specific Heat')
    plt.legend()
    plt.show()
    # Plot
    plt.figure()
    plot = plot_data
    for n, c_data in plot.items():
        plt.plot(c_data, label=f'n={n}')
    plt.title('Specific Heat per Spin ($C/N$) vs. Temperature ($T$)')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat per Spin')
    plt.legend(); plt.grid(True)
    plt.show()
    return plot_data, c_max_per_N_list, n_list


if __name__=='__main__':
    argv
    arg = argv[1][7]
    if arg=='1':
        print('aaa')
        y = []
        x = np.linspace(1.5, 5, 50)
        for i in x:
            print(i)
            state = init()
            state, m, _, _, _ = step(state, t=i, st=25000)
            #plt.matshow(state)
            print(m)
            y += [m]
        plt.plot(x, y)
        plt.xlabel('Temperature')
        plt.ylabel('Magnetization')
        plt.show()
    if arg=='2':
        plot_data, c_max_per_N_list, n_list = part2()