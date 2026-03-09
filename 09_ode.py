import numpy as np
import matplotlib.pyplot as plt

def f1(y, t):
    """ Pravá strana ODR pro testovací nehomogenní neautonomní rovnici 1. řádu. """
    return np.sin(t * y)

def f_lorenz(Y, t, sigma, rho, beta):
    """ Pravá strana soustavy ODR pro Lorenzův systém."""
    x, y, z = Y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return np.array([dxdt, dydt, dzdt])

def euler_1_step(f, y, t, dt):
    """ Jeden krok Eulerovy metody 1. řádu."""
    return y + f(y, t) * dt

def euler_2_step(f, y, t, dt):
    """ Jeden krok Eulerovy metody 2. řádu."""
    k1 = f(y, t)
    k2 = f(y + k1 * dt, t + dt)
    return y + (k1 + k2) * dt / 2

def ode_solver(f, initial_condition, dt, t_min, t_max, step_function=euler_1_step):
    """ Obecný ODE solver pro řešení ODR pomocí explicitního algoritmu zadané integrační metody. """
    t = t_min
    y = initial_condition

    t_result = [t]
    y_result = [y]

    while t < t_max:
        y = step_function(f, y, t, dt)
        t += dt

        t_result.append(t)
        y_result.append(y)

    return np.array(t_result), np.array(y_result)

if __name__ == "__main__":
    # Testovací ODR 1. řádu
    initial_condition = np.array([1.0])
    
    dt = 0.1

    t_min = 0.0
    t_max = 10.0

    t1, y1 = ode_solver(f1, initial_condition, dt, t_min, t_max, step_function=euler_1_step)
    t2, y2 = ode_solver(f1, initial_condition, dt, t_min, t_max, step_function=euler_2_step)

    plt.plot(t1, y1, label=f'Euler 1. řádu pro dt = {dt}')
    plt.plot(t2, y2, label=f'Euler 2. řádu pro dt = {dt}')

    plt.xlabel('t')
    plt.ylabel('y(t)')

    plt.legend()
    plt.grid()
    plt.show()

    # Lorenzův systém - citlivost na počáteční podmínky
    initial_condition1 = np.array([1, 1, 1])
    initial_condition2 = np.array([1.000000001, 1, 1])

    dt = 0.001
    t_min = 0.0
    t_max = 100.0

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    t1, y1 = ode_solver(lambda Y, t: f_lorenz(Y, t, sigma, rho, beta), initial_condition1, dt, t_min, t_max, step_function=euler_2_step)
    t2, y2 = ode_solver(lambda Y, t: f_lorenz(Y, t, sigma, rho, beta), initial_condition2, dt, t_min, t_max, step_function=euler_2_step)

    plt.plot(t1, y1[:, 0], label=f'dt = {dt}, Euler 2. řádu')
    plt.plot(t2, y2[:, 0], label=f'dt = {dt}, Euler 2. řádu s poruchou v počáteční podmínce')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.show()

    # Lorenzův atraktor
    plt.plot(y1[:, 0], y1[:, 2], label='Lorenzův atraktor')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()
    plt.grid()
    plt.show()