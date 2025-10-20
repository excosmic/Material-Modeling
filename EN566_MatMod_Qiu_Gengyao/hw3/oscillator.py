import numpy as np
import matplotlib.pyplot as plt
import sys
sin = np.sin


delta_t = 0.1 # s
g = 9.8 # m/s2
l = 9.8 # m
alpha = 0.2 # rad/s2
gamma = 0.25 # s-1
Omega = 0.935 # rad/s

def euler(Omega=0.935, t_all=80, non_linear_effect=lambda x:x, theta_in=(0.0, 0.0)):
    theta, v = [theta_in[0]], [theta_in[1]]
    t = 0.0
    while t<t_all:
        a = -g/l*non_linear_effect(theta[-1])-2*gamma*v[-1]+alpha*sin(Omega*t)
        v += [v[-1]+a*delta_t]
        theta += [theta[-1]+v[-1]*delta_t]
        t += delta_t
    return np.array([theta, v]).T

def kutta(Omega=0.935, t_all=80, non_linear_effect=lambda x:x, theta_in=(0.0, 0.0)):
    f = lambda t, theta1, theta2: -g/l*non_linear_effect(theta1)-2*gamma*theta2+alpha*sin(Omega*t)
    F = lambda t, theta: np.array([theta[1], f(t, theta1=theta[0], theta2=theta[1])])
    # Y have delta_t step width
    Y = [np.array([theta_in[0], theta_in[1]])] # first is theta and second is theta'
    debug_sin = []
    t = 0.0
    while t<t_all:
        k1 = F(t,           Y[-1])
        k2 = F(t+delta_t/2, Y[-1]+delta_t/2*k1)
        k3 = F(t+delta_t/2, Y[-1]+delta_t/2*k2)
        k4 = F(t+delta_t,   Y[-1]+delta_t*k3)
        Y += [Y[-1]+delta_t/6*(k1+2*k2+2*k3+k4)]
        debug_sin += [sin(Omega*t)]
        t+=delta_t
        pass
    Y = np.array(Y)
    return Y

def amplitude_phase(t_all=40, omega_range=[0.0001, 2.5], method=kutta):
    ans = [] # where [:, 0] is omega, [:. 1] is amplitude and [:, 2] is phase shift
    for omega in np.linspace(omega_range[0], omega_range[1], 300):
        y = method(Omega=omega, t_all=t_all)
        y_stable = y[len(y)//2:] # assume that this curve will come to stable after half of length.
        t_stable = np.array([i*delta_t+t_all/2 for i in range(len(y_stable))])
        # linear regression
        X = np.array([np.sin(omega*t_stable), np.cos(omega*t_stable)]).T # design matrix
        beta_hat = np.linalg.inv(X.T@X)@X.T@(y_stable[:, 0].reshape(-1, 1))
        # calculate amplitude and phase shift
        my_arctan = lambda a, b: np.arctan(a/b) if b>0 else (np.pi+np.arctan(a/b) if a>=0 else -np.pi+np.arctan(a/b))
        ans += [[omega, (beta_hat[0][0]**2+beta_hat[1][0]**2)**0.5, np.abs(my_arctan(beta_hat[1][0], beta_hat[0][0]))]]
    ans = np.array(ans)
    # verify (calculate the FWHM)
    peak_idx = np.argmax(ans[:, 1])
    half_max = ans[peak_idx, 1]/2
    left_omega = ans[:peak_idx, 0]
    right_omega = ans[peak_idx:, 0]
    omega1 = left_omega[np.argmin(np.abs(ans[:peak_idx, 1]-half_max))]
    omega2 = right_omega[np.argmin(np.abs(ans[peak_idx:, 1]-half_max))]
    print(f'FWHM = {omega2-omega1}')
    """plt.plot(ans[:, 0], ans[:, 1], label='amp')
    plt.plot(ans[:, 0], ans[:, 2], label='phase')
    plt.legend(); plt.grid(); plt.show()"""
    return ans[:, 0], ans[:, 1], ans[:, 2], omega2-omega1 # x, amplitude, phase_shift, FWHM

def potential_kinetic_total():
    Y = kutta()
    potential = lambda theta:(l - l*np.cos(theta))*g
    kinetic = lambda theta1:0.5*(theta1*l)**2
    p = potential(Y[:, 0])
    k = kinetic(Y[:, 1])
    return p, k, p+k # potentoal energy kinetic energy and total energy

def add_non_linear(method=kutta):
    global alpha
    alpha = 1.2
    y_origin = kutta()
    y = kutta(non_linear_effect=lambda x:np.sin(x))
    #plt.plot(y[:, 0], label="non linear")
    #plt.plot(y_origin[:, 0], label="linear")
    #plt.title("add non linear"); plt.xlabel('step'); plt.ylabel('theta')
    #plt.legend(); plt.grid(); plt.show()
    return y, y_origin

def compute_delta_theta():
    global alpha
    for a in [0.2, 0.5, 1.2]:
        alpha = a
        y1 = kutta(Omega=0.666, non_linear_effect=lambda x:np.sin(x), theta_in=[0.0, 0.0])
        y2 = kutta(Omega=0.666, non_linear_effect=lambda x:np.sin(x), theta_in=[0.001, 0.0])
        x = [delta_t*i for i in range(len(y1))]
        plt.plot(x, logy:=np.log(np.abs(y2[:,0]-y1[:,0])), label=f'alpha_D={a}')
        # linear regression
        X = np.array([np.ones(len(logy)), x]).T # design matrix
        beta_hat = np.linalg.inv(X.T@X)@X.T@(logy.reshape(-1, 1))
        print(f"lambda of while alpha_D={a} is {beta_hat[1]}")
    plt.title("Several Trajectories With Slightly Different Initial Angle")
    plt.xlabel("time")
    plt.ylabel("trajectories(log10)")
    plt.grid(); plt.legend(); plt.show()
    return 
# test & debug
if False:
    Y, debug_sin = kutta(Omega=0.01, t_all=400)
    plt.plot(Y[:, 0])
    plt.plot(debug_sin)
    plt.show()
    print(Y, debug_sin)

# Test the linear regression
if False:
    t_all = 40
    omega = 0.01
    y = kutta(Omega = omega)
    y_stable = y[len(y)//2:]
    t_stable = np.array([i*delta_t+t_all for i in range(len(y_stable))])
    # linear regression
    X = np.array([np.sin(omega*t_stable), np.cos(omega*t_stable)]).T# design matrix
    beta_hat = np.linalg.inv(X.T@X)@X.T@(y_stable[:, 0].reshape(-1, 1))
    y_regressed = beta_hat[0][0]*np.sin(omega*t_stable)+beta_hat[1][0]*np.cos(omega*t_stable)
    print(y_regressed, y_stable[:, 0])
    print(beta_hat[0][0], beta_hat[1, 0], np.arctan(beta_hat[1][0]/beta_hat[0][0]))

def get_part():
    arg = sys.argv[1][7:]
    arg = int(arg)
    return arg

if __name__=='__main__':
    match get_part():
        case 1:
            print('Please see the analytical answers in PDF.')
        case 2:
            x_e, amp_e, shift_e, fwhm_e = amplitude_phase(method=euler)
            x_k, amp_k, shift_k, fwhm_e = amplitude_phase(method=kutta)
            fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
            axes[0][0].plot(x_e, amp_e, label='amplitude'); axes[0][1].plot(x_e, shift_e, label='phase_shift');
            axes[1][0].plot(x_k, amp_e, label='amplitude'); axes[1][1].plot(x_k, shift_k, label='phase_shift');
            axes[0][0].set_title('Amplitude by Using Euler Method'); axes[0][0].set_xlabel('Omega_d'); axes[0][0].set_ylabel('Amplitude')
            axes[1][0].set_title('Amplitude by Using RK4 Method'); axes[1][0].set_xlabel('Omega_d'); axes[1][0].set_ylabel('Amplitude')
            axes[0][1].set_title('Phase Shift by Using Euler Method'); axes[0][1].set_xlabel('Omega_d'); axes[0][1].set_ylabel('Phase Shift')
            axes[1][1].set_title('Phase Shift by Using RK4 Method'); axes[1][1].set_xlabel('Omega_d'); axes[1][1].set_ylabel('Phase Shift')
            plt.show()
        case 3:
            p, k, t = potential_kinetic_total()
            plt.plot(p, label='Potential'); plt.plot(k, label='Kinetic'); plt.plot(p+k, label='Total')
            plt.legend(); plt.show()
        case 4:
            kutta_y, kutta_y_origin = add_non_linear()
            euler_y, euler_y_origin = add_non_linear(method=euler)
            x = np.array([i*delta_t for i in range(len(kutta_y))])
            fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
            axes[0][0].plot(x, kutta_y[:, 0], label='linear'); axes[0][0].plot(x, kutta_y_origin[:, 0], label='none linear')
            axes[0][1].plot(x, euler_y[:, 0], label='linear'); axes[0][1].plot(x, euler_y_origin[:, 0], label='none linear')
            axes[1][0].plot(x, kutta_y[:, 1], label='linear'); axes[1][0].plot(x, kutta_y_origin[:, 1], label='none linear')
            axes[1][1].plot(x, euler_y[:, 1], label='linear'); axes[1][1].plot(x, euler_y_origin[:, 1], label='none linear')
            axes[0][0].set_title('In RK4 Method'); axes[0][0].set_xlabel('Time'); axes[0][0].set_ylabel('theta')
            axes[0][1].set_title('In Euler Method'); axes[0][1].set_xlabel('Time'); axes[0][1].set_ylabel('theta')
            axes[1][0].set_title('In RK4 Mothod'); axes[1][0].set_xlabel('Time'); axes[1][0].set_ylabel('omega')
            axes[1][1].set_title('In Euler Mothod'); axes[1][1].set_xlabel('Time'); axes[1][1].set_ylabel('omega')
            plt.legend();plt.show()
        case 5:
            compute_delta_theta()