from typing import List, Set, Dict, Tuple, Optional, Any, Union
import numpy as np
from scipy.integrate import quad
from scipy.stats import chi
import pandas as pd

def rateSinu(x, val,freq,phase):
    return 2*val, val*np.cos(freq*2*np.pi*x + phase/180*np.pi) + val

def rateSinu2(x, val,freq,phase):
    return 4.6*val, val*np.cos(freq*2*np.pi*x + phase/180*np.pi) + 1.3*val*np.cos(1.3*freq*2*np.pi*x + phase/180*np.pi) + 2.3*val

def rateSquare(x, val,freq, phase, offset=120000, duty = 0.5):
    a = (x * freq + phase) - np.floor(x*freq + phase);
    b = a<=duty;
    return 0.2*val + offset , 0.1*val * b * 2 + offset

def generateStamps(val, freq, phase, rate_function, T):
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
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def generateStampsGeneral(rate, ts=None, T=1):
    t = 0
    timestamps = []
    lambda_u = np.max(rate)
    while t<T:
        u1 = np.random.uniform()
        t -= 1/lambda_u*np.log(u1)
        u2 = np.random.uniform()
        val2 = np.interp(t, ts, rate)
        if u2 <= val2/ lambda_u:
            timestamps.append(find_nearest(ts, t))
    return np.array(timestamps)

def generateStamps_withDeadTime(val, freq, phase, rate_function, dead_time, T):
    t = 0
    timestamps = []
    lambda_u, c= rate_function(0,val,freq,phase)
    while t<T:
        u1 = np.random.uniform()
        t -= 1/lambda_u*np.log(u1)
        u2 = rng.uniform()
        c, val2 = rate_function(t,val,freq,phase)
        if u2 <= val2/ lambda_u:
            timestamps.append(t)
            t += dead_time
        
    return np.array(timestamps)

def applyDeadtime(stamps,deadtime,jitter):
    pruned_stamps = []
    old_t = -np.inf
    for t in stamps:
        t = t + jitter*np.random.rand(1)
        if t - old_t > deadtime:
            pruned_stamps.append(t)
            old_t = t
    return np.array(pruned_stamps)

def func_wrap(x, rate, times):
    return np.interp(x, times, rate)

def rateIntegral(rate, times):
    integral = [quad(func_wrap, times[i], times[i+1], args=(rate, times))[0] for i in range(times.shape[0]-1)]
    return np.cumsum([0] + integral).copy()

def generateStampsCinlar(val, freq, phase, rate_function, T):
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

print('Done')

def findScale(a,listVal):
    counter = 1
    for cur_val in listVal[::-1]:
        if a >= cur_val:
            break
        counter = counter+1
    return len(listVal) - counter
    
    
def rate_fun_integ(fun, xs):
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
