from typing import List, Set, Dict, Tuple, Optional, Any, Union
import numpy as np
from scipy.integrate import quad
from scipy.stats import chi
import pandas as pd

def rateSinu(x, val,freq,phase):
    """
    A sinusoidal rate function.

    @param x: float
        The time or input value.
    @param val: float
        The amplitude of the rate function. In this case also the offset (Note intensity > 0)
    @param freq: float
        The frequency of the oscillation in cycles per second (Hz).
    @param phase: float
        The phase shift of the oscillation (degrees).

    @return lambda_u: float
        The constant upper bound on the rate (2 * val).
    @return rate: float
        The value of the sinusoidal rate function at time x.
        It is calculated as val * cos(2 * pi * freq * x + phase in radians) + val.
    """
    return 2*val, val*np.cos(freq*2*np.pi*x + phase/180*np.pi) + val

def rateSinu2(x, val,freq,phase):
    """
    A more complex sinusoidal rate function with two frequency components.

    @param x: float
        The time or input value.
    @param val: float
        A amplitude or scaling factor of the rate function.
    @param freq: float
        The base frequency of the oscillation in cycles per second (Hz).
    @param phase: float
        The phase shift of the oscillation in degrees.

    @return lambda_u: float
        The ouput has constant upper bound (4.6 * val).
    @return rate: float
        The value of the rate function at time x. 
        This is a combination of two cosine waves:
        - First cosine with frequency `freq` and amplitude `val`.
        - Second cosine with frequency 1.3 * `freq` and amplitude 1.3 * `val`.
        An additional constant offset 2.3 * `val` is added.
    """
    return 4.6*val, val*np.cos(freq*2*np.pi*x + phase/180*np.pi) + 1.3*val*np.cos(1.3*freq*2*np.pi*x + phase/180*np.pi) + 2.3*val

def rateSquare(x, val,freq, phase, offset=120000, duty = 0.5):
    """
    A square wave rate function with a specified duty cycle and offset.

    @param x: float
        The time or input value.
    @param val: float
        The amplitude or scaling factor of the rate function.
    @param freq: float
        The frequency of the square wave in cycles per second (Hz).
    @param phase: float
        The phase shift of the square wave.
    @param offset: float, optional
        A constant offset added to the output rate (default is 120000).
    @param duty: float, optional
        The duty cycle of the square wave, i.e., the fraction of the cycle for which the output is 'high' (default is 0.5).

    @return lambda_u: float
        The upper bound on the output function (0.2 * val + offset).
    @return rate: float
        The value of the square wave rate function at time x, adjusted by the offset and duty cycle.
    """
    a = (x * freq + phase) - np.floor(x*freq + phase);
    b = a<=duty;
    return 0.2*val + offset , 0.1*val * b * 2 + offset

def generateStamps(val, freq, phase, rate_function, T):
    """
    Generates event timestamps using a non-homogeneous Poisson process, based on a specified rate function.

    @param val: float
        The amplitude or scaling factor for the rate function.
    @param freq: float
        The frequency parameter passed to the rate function.
    @param phase: float
        The phase parameter passed to the rate function.
    @param rate_function: function
        A callable rate function (e.g., rateSinu, rateSinu2, or rateSquare) that defines the rate of events over time.
    @param T: float
        The total time duration for which to generate events.

    @return timestamps: np.array
        An array of timestamps where events occurred, based on the rate function.

    Process:
    - The function simulates event times by iterating through small intervals of time:
        - In a Poisson processm, inter-arrival time between events are exponentially distributed.
        - To generate timestamps, the function uses rejection sampling:
        1. First generate a candidate event t using the exponential distribution with rate `lambda_u` (upper bound of the rate function).
            t -= (1 / lambda_u) * log(u1)
        2. Accept / reject step:
        if u2 <= rateFunc(t) / lambda_u: accept the event at time t. and move to generate next event.
    """
    t = 0
    timestamps = []
    lambda_u, c= rate_function(0,val,freq,phase)
    while t<T:
        u1 = np.random.uniform()
        t -= 1/lambda_u*np.log(u1)
        u2 = np.random.uniform()
        c, val2 = rate_function(t,val,freq,phase)
        if u2 <= val2/ lambda_u:
            timestamps.append(t)
    return np.array(timestamps)

def find_nearest(array, value):
    """
    @param array: np.array
        The array of values to search through.
    @param value: float
        The target value for which the nearest element is sought.

    @return nearest_value: float
        The value in the array that is closest to the input `value`.
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def generateStampsGeneral(rate, ts=None, T=1):
    """
    Generates event timestamps based on a general rate function, using rejection sampling.

    @param rate: np.array
        Array representing the function at each corresponding time in `ts`. Does not have to be periodic
    @param ts: np.array, optional
        Array of time points corresponding to the rate values. If None, defaults to a linear time grid.
    @param T: float, optional
        The total time duration for generating events (default is 1).

    @return timestamps: np.array
        An array of timestamps where events occurred, based on the rate function.

    Explanation:
    - The function generates candidate event times using the upper bound `lambda_u` on the rate.
    - It uses rejection sampling to accept or reject candidate times based on the actual rate at the candidate time.
    - The function interpolates the rate at non-grid times using `np.interp`.
    """
    t = 0
    timestamps = []
    lambda_u = np.max(rate)
    while t<T:
        u1 = np.random.uniform()
        t -= 1/lambda_u*np.log(u1)
        u2 = np.random.uniform()
        val2 = np.interp(t, ts, rate) # Piecewise linear interpolation
        if u2 <= val2/ lambda_u:
            timestamps.append(find_nearest(ts, t))
    return np.array(timestamps)

def generateStamps_withDeadTime(val, freq, phase, rate_function, dead_time, T):
    """ Not used """
    t = 0
    timestamps = []
    lambda_u, c= rate_function(0,val,freq,phase)
    while t<T:
        u1 = np.random.uniform()
        t -= 1/lambda_u*np.log(u1)
        u2 = np.random.uniform()
        c, val2 = rate_function(t,val,freq,phase)
        if u2 <= val2/ lambda_u:
            timestamps.append(t)
            t += dead_time # Deadtime is enforced by skipping the next `dead_time` interval
        
    return np.array(timestamps)

def applyDeadtime(stamps,deadtime,jitter):
    """
    Applies SPAD dead time and jitter to an array of event timestamps.

    @param stamps: np.array
        An array of event timestamps.
    @param deadtime: float
        SPAD deadtime.
    @param jitter: float
        A random noise added to each event time to simulate jitter.

    @return : np.array
        An array of timestamps with dead time and jitter applied.
    """
    pruned_stamps = []
    old_t = -np.inf
    for t in stamps:
        t = t + jitter*np.random.rand(1)
        if t - old_t > deadtime:
            pruned_stamps.append(t)
            old_t = t
    return np.array(pruned_stamps)

def func_wrap(x, rate, times):
    """
    Wrapper function for interpolation.

    @param x: float
        The point at which to evaluate the interpolated rate.
    @param rate: np.array
        Array representing the rate function.
    @param times: np.array
        Array of time points corresponding to the rate function values.

    @returns: float
        The interpolated rate at the point `x`.
    """
    return np.interp(x, times, rate)

def rateIntegral(rate, times):
    """
    Computes the cumulative integral of the interpolated rate function over time.

    @param rate: np.array
        Array representing the rate function values.
    @param times: np.array
        Array of time points corresponding to the rate values.

    @returns: An array of cumulative integral values at each time point. 0 at time 0.
    """
    integral = [quad(func_wrap, times[i], times[i+1], args=(rate, times))[0] for i in range(times.shape[0]-1)]
    return np.cumsum([0] + integral).copy()

def generateStampsCinlar(val, freq, phase, rate_function, T):
    """
    Seems to still be under experimentation. TODO
    """
    def myfunc(val,freq,phase, mt):
        return lambda a : val*a + val*np.sin(2*np.pi*freq*a + phase/180*np.pi)/(2*np.pi*freq) - mt

    s = 0
    t = 0
    timestamps = []
    while t < T:
        u = np.random.uniform()
        s = s - np.log(u)
        
        newFunc = myfunc(val,freq,phase, s)
        t = invertSin(newFunc, s/val - 1/freq, s/val + 1/freq, tol=1e-15, max_iter = 100)
        #s_estimated = val*t + val*np.sin(2*np.pi*freq*t + phase)/(2*np.pi*freq)
        #print(s,s_estimated)
        if t < T:
            timestamps.append(t)
    return np.array(timestamps).copy()

def invertSin(f,a,b,tol=1e-7, max_iter=10000):
    """
    Inverts a sine function using the bisection fixed-point method. ie solve x = sin(x)
    Should be able to be used on any function that has root in the interval [a,b].

    @param f: function
        The function to invert.
    @param a: float
        The lower bound of the interval.
    @param b: float
        The upper bound of the interval.
    @param tol: float, optional
        The tolerance for convergence (default is 1e-7).
    @param max_iter: int, optional
        The maximum number of iterations allowed (default is 10000).

    @return root: float
        The root of the function within the specified interval.
    """
    for n in range(max_iter):
        c = (a+b)/2
        if (f(c) == 0) or ((b-a)/2 < tol):
            return c
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    #print('Method failed')
    return c

def findScale(a,listVal):
    """
    Finds the position of `a` in a list by counting how many values are larger.
    TODO: What is this used for???
    """
    counter = 1
    for cur_val in listVal[::-1]:
        if a >= cur_val:
            break
        counter = counter+1
    return len(listVal) - counter
    
    
def rate_fun_integ(fun, xs):
    """
    Wrappper around quad intergral. Doesn't seem to be used to useful? TODO
    """
    output = np.zeros_like(xs)
    for id, x in enumerate(xs):
        output[id], _ = quad(fun, 0, x)
    return output

def compute_amplitude_bound(percentage_in, counts_per_sec):
    """
    Computes the bound value based on the chi distribution of the estimated amplitudes.
    :param percentage_in:
    :param counts_per_sec: Normalised counts per second, e.g  timestamps.size / timestamps[-1] / exposure
    :return:
        bound_val: the estimated bound
    """

    alpha_1 = chi.ppf(percentage_in, 2)
    bound_val = alpha_1 * np.sqrt(counts_per_sec * 2)
    return bound_val
