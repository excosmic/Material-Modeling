import sys
import matplotlib.pyplot as plt
import numpy as np

def print_img(pa, fig, title:str):
    if 'print' in pa.keys():
        for fmt in pa['print']:
            if fmt!='': fig.savefig(f'{title}.{fmt}', dpi=fig.dpi)
    return

def main():
    # parse the arguments
    pa:dict = {}
    for arg in sys.argv:
        match arg[2]:
            case 'f': pa['function'] = arg[11:].split(',') # list[str]
            case 'w': pa['write'] = arg[8:] # str
            case 'r': pa['read'] = arg[17:] # str
            case 'p': pa['print'] = arg[8:].split(',')
    #
    if 'read' in pa.keys():
        file_name = pa['read']
        read_mat = np.loadtxt(file_name)
        fig = plt.figure()
        plt.xlabel('x'); plt.ylabel('y'); j=1
        for i in (read_mat.T)[1:]:
            print(i)
            plt.plot((read_mat.T)[0], i, label=f'column{j}');j+=1
        plt.legend(); plt.title('Data From File'); plt.show()
        print_img(pa, fig, 'read_data')
        return
    if 'function' in pa.keys():
        fig = plt.figure()
        x = np.arange(-10, 10.05, 0.05)
        write_list:list = [list(x)]
        plt.xlabel('x')
        plt.ylabel('y')
        for f in pa['function']:
            if f=='cos': plt.plot(x, j:=np.cos(x), label='y=cos(x)')
            if f=='sin': plt.plot(x, j:=np.sin(x), label='y=sin(x)')
            if f=='sinc': plt.plot(x, j:=np.sinc(x), label='y=sinc(x)')
            if f!='': write_list += [list(j)]
        plt.legend()
        plt.title('Trigonometry')
        print_img(pa, fig, 'trigonometry')
        if 'write' in pa.keys():
            write_mat = np.array(write_list).T
            np.savetxt(pa['write'], write_mat)
        plt.show()
    return

if __name__=='__main__': main()