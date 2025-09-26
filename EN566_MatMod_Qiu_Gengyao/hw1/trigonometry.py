import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    # parse the arguments
    pa:dict = {}
    for arg in sys.argv:
        match arg[2]:
            case 'f': pa['function'] = arg[11:].split(',') # list[str]
            case 'w': pa['write'] = arg[8:] # str
            case 'r': pa['read'] = arg[17:] # str
            case 'p': pa['print'] = arg[8:].split(',')
    # --function is required and other is optional.
    # prepare the plot canvas.
    if 'read' in pa.keys():
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
    # draw the function
    if 'function' in pa.keys():
        x = np.arange(-10, 10.05, 0.05)
        write_list = [list(x)]
        ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        for f in pa['function']:
            if f=='cos': ax1.plot(x, j:=np.cos(x), label='y=cos(x)')
            if f=='sin': ax1.plot(x, j:=np.sin(x), label='y=sin(x)')
            if f=='sinc':ax1.plot(x, j:=np.sinc(x),label='y=sinc(x)')
            if f!='':write_list +=[list(j)]
        if 'write' in pa.keys():np.savetxt(pa['write'], np.array(write_list).T)
        ax1.legend()
        ax1.set_title('Functions From function Argument')
    if 'read' in pa.keys():
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); j=1
        for i in (mat:=np.loadtxt(pa['read']).T)[1:]:
            ax2.plot(mat[0], i, label=f'column{j}');j+=1
        ax2.legend();ax2.set_title('Data from read_from_file Argument')
    if 'print' in pa.keys():
        for fmt in pa['print']:
            if fmt!='': plt.savefig(f'print_img.{fmt}', dpi=fig.dpi)
    plt.show()

if __name__=='__main__':main()