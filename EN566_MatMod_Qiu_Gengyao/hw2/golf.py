import numpy as np
import matplotlib.pyplot as plt
import sys
m = 46e-3 #kg
v0 = 70 #m/s
rho= 1.29 #kg/m^3
g = 9.8 # m/s^2
A = 0.0014 # m^2


def ideal(theta_degree:float, delta_t=0.1):
    theta_rad = theta_degree/180*np.pi
    x, y = [0.0], [0.0]
    vx, vy = [v0*np.cos(theta_rad)], [v0*np.sin(theta_rad)]
    while y[-1]>=0:
        x += [x[-1] + vx[-1]*delta_t]
        vx += [vx[-1]]
        y += [y[-1] + vy[-1]*delta_t]
        vy += [vy[-1] - g*delta_t]
    return x, y

def smooth(theta_degree:float, c=1/2, delta_t=0.1):
    theta_rad = theta_degree/180*np.pi
    x, y = [0.0], [0.0]
    vx, vy = [v0*np.cos(theta_rad)], [v0*np.sin(theta_rad)]
    while y[-1]>=0:
        x += [x[-1]+vx[-1]*delta_t]
        v_norm = (vx[-1]**2+vy[-1]**2)**0.5
        vx += [vx[-1]-A*v_norm/m*vx[-1]*delta_t]
        y += [y[-1]+vy[-1]*delta_t]
        vy += [vy[-1]-g*delta_t-A*v_norm/m*vy[-1]*delta_t]
    return x, y

def drag(theta_degree:float, c=lambda v:1/2, delta_t=0.1):
    theta_rad = theta_degree/180*np.pi
    x, y = [0.0], [0.0]
    vx, vy = [v0*np.cos(theta_rad)], [v0*np.sin(theta_rad)]
    while y[-1]>=0:
        norm = (vx[-1]**2+vy[-1]**2)
        x += [x[-1]+vx[-1]*delta_t]
        vx += [vx[-1]-c(norm**0.5)*rho*A*norm/m*np.cos(theta_rad)*delta_t]
        y += [y[-1]+vy[-1]*delta_t]
        vy += [vy[-1]-(g+c(norm**0.5)*rho*A*norm/m*np.sin(theta_rad))*delta_t]
    return x, y

def spin(theta_degree:float, S0omega=0.25*m, c=lambda v:1/2 if v<=14 else 7.0/v, delta_t=0.1):
    theta_rad = theta_degree/180*np.pi
    x, y = [0.0], [0.0]
    vx, vy = [v0*np.cos(theta_rad)], [v0*np.sin(theta_rad)]
    while y[-1]>=0:
        norm = vx[-1]**2+vy[-1]**2
        x += [x[-1]+vx[-1]*delta_t]
        vx +=[vx[-1]-(c(norm**0.5)*rho*A*norm/m*np.cos(theta_rad)+S0omega/m*vy[-1])*delta_t]
        y += [y[-1]+vy[-1]*delta_t]
        vy += [vy[-1]-(g+c(norm**0.5)*rho*A*norm/m*np.sin(theta_rad)-S0omega/m*vx[-1])*delta_t]
    return x[:-1], y[:-1]


if __name__=='__main__':
    if sys.argv[1][:7]!='--plot=':raise Exception('Expect --plot=')
    theta_degree = float(sys.argv[1][7:])
    plt.plot(*ideal(theta_degree, delta_t=0.1), label='ideal')
    plt.plot(*drag(theta_degree, delta_t=0.1), label='smooth')
    plt.plot(*drag(theta_degree, c=lambda v:0.5 if v<=14 else 7.0/v, delta_t=0.1), label='dimpled')
    plt.plot(*spin(theta_degree, delta_t=0.1), label='dimpled+spin')
    plt.legend()
    plt.title(f'Trajectories of golf ball in {theta_degree} degree')
    plt.xlabel('Distance')
    plt.ylabel('Height')
    plt.show()