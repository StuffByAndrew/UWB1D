import numpy as np
import time
import tqdm
import finufft
from scipy.optimize import bisect
import scipy.signal as ss

from scipy.stats import chi
import torch

def probe_frequencies_sweep(stamps, time_total, freqs, verbose=True, return_freqs=False):
    """
    Sweeps across all the provided frequencies.
    :param time_total: exposure time in seconds
    :param freq: either a tuple (indicating start, stop, step) or a number
    :param stamps: 1D numpy containing the timestamps.
    :param verbose: whether to display estimated time of completion or not
    :return:
        f : 1D numpy array
    """
    if isinstance(time_total, int) or isinstance(time_total, float):
        stamps = stamps[stamps < time_total]
    elif len(time_total) == 2:
        stamps = stamps[(stamps >= time_total[0])*(stamps < time_total[1])] - time_total[0]
        time_total = time_total[1] - time_total[0]
    if verbose:
        tic = time.time()
    if isinstance(freqs, int):
        f = finufft.nufft1d1(np.mod(stamps*2*np.pi, 2*np.pi), np.ones_like(stamps).astype('complex128'), 2*int(freqs)-1)/time_total
        f = f[int(freqs)-1:]
        f[1:]*=2
    elif isinstance(freqs, np.ndarray):
        f = finufft.nufft1d3(stamps*2*np.pi /time_total, (np.ones_like(stamps)).astype('complex128'), freqs*time_total) / time_total
        f[freqs != 0]*=2
    elif len(freqs) == 3:
        #This is the old version, that uses type-3 FINUFFT
        #probed_freqs = np.arange(freqs[0], freqs[1] + freqs[2], freqs[2])
        #f = finufft.nufft1d3(np.mod(stamps*2*np.pi * freqs[2], 2*np.pi), (np.ones_like(stamps)).astype('complex128'), probed_freqs/freqs[2]) / time_total
        probed_freqs = np.arange(0, freqs[1] + freqs[2] - freqs[0], freqs[2])
        N = len(probed_freqs)
        f = finufft.nufft1d1(np.mod(stamps*2*np.pi * freqs[2], 2*np.pi), np.exp(2j*np.pi*freqs[0] * stamps), 2*N-1) / time_total
        f = f[N-1:]
        f[(probed_freqs + freqs[0]) != 0]*=2
    else:
        #Do some error stuff
        raise Exception("freqs either needs to be a 3-tuple or a integer")

    if verbose:
        print(f'Elapsed time: {time.time() - tic:03.3f} seconds')

    if return_freqs == False:
        return f
    
    else:
        return f, probed_freqs + freqs[0]

def probe_frequencies_sweep_windowed(stamps, time_total, freqs, window_func = None, verbose=True, return_freqs=False):
    """
    Sweeps across all the provided frequencies.
    :param time_total: exposure time in seconds
    :param freq: either a tuple (indicating start, stop, step) or a number
    :param stamps: 1D numpy containing the timestamps.
    :param verbose: whether to display estimated time of completion or not
    :return:
        f : 1D numpy array
    """
    if window_func == None:
        window_func = lambda x: 0.5 - 0.5*np.cos(2*np.pi*x /time_total)
    if isinstance(time_total, int) or isinstance(time_total, float):
        stamps = stamps[stamps < time_total]
    elif len(time_total) == 2:
        stamps = stamps[(stamps >= time_total[0])*(stamps < time_total[1])] - time_total[0]
        time_total = time_total[1] - time_total[0]
    if verbose:
        tic = time.time()
    if isinstance(freqs, int):
        f = finufft.nufft1d1(np.mod(stamps*2*np.pi, 2*np.pi), window_func(stamps).astype('complex128'), 2*int(freqs)-1)/time_total
        f = f[int(freqs)-1:]
        f[1:]*=2
    elif isinstance(freqs, np.ndarray):
        f = finufft.nufft1d3(stamps*2*np.pi /time_total, window_func(stamps).astype('complex128'), freqs*time_total) / time_total
        f[freqs != 0]*=2
    elif len(freqs) == 3:
        #This is the old version, that uses type-3 FINUFFT
        #probed_freqs = np.arange(freqs[0], freqs[1] + freqs[2], freqs[2])
        #f = finufft.nufft1d3(np.mod(stamps*2*np.pi * freqs[2], 2*np.pi), (np.ones_like(stamps)).astype('complex128'), probed_freqs/freqs[2]) / time_total
        probed_freqs = np.arange(0, freqs[1] + freqs[2] - freqs[0], freqs[2])
        N = len(probed_freqs)
        f = finufft.nufft1d1(np.mod(stamps*2*np.pi * freqs[2], 2*np.pi), np.exp(2j*np.pi*freqs[0] * stamps) * window_func(stamps), 2*N-1) / time_total
        f = f[N-1:]
        f[(probed_freqs + freqs[0]) != 0]*=2
    else:
        #Do some error stuff
        raise Exception("freqs either needs to be a 3-tuple or a integer")

    if verbose:
        print(f'Elapsed time: {time.time() - tic:03.3f} seconds')

    if return_freqs == False:
        return f
    
    else:
        return f, probed_freqs + freqs[0]
    
def extract_amp_phase(f):
    """
    Returns the amplitudes and phases of f
    :param f: np array of complex variables
    """
    amplitudes = np.abs(f)
    phases = np.arctan2(np.imag(f), np.real(f))
    return amplitudes,phases

def compute_amplitude_bound(stamps, time_total, percentage_in):
    """
    Computes the bound value based on the chi distribution of the estimated amplitudes.
    :param percentage_in:
    :param counts_per_sec: Normalised counts per second, e.g  timestamps.size / timestamps[-1] / exposure
    :return:
        bound_val: the estimated bound
    """
    counts_per_sec = stamps.shape[0]/stamps[-1]/time_total
    alpha_1 = chi.ppf(percentage_in, 2)
    bound_val = alpha_1 * np.sqrt(counts_per_sec * 2)
    return bound_val

def reconstruct_rate_function(ts, freqs, f, verbose = True):
    """
    Reconstructs the 1D rate function of a non-homohegeneous Poisson (NHPP) distribution.
    :param ts: 1D tensor of the time values
    :param frequencies: 1D tensor of frequencies used for the reconstruction
    :param amplitudes: 1D tensor of the corresponding amplitudes of the frequencies
    :param phases: 1D tensor of the corresponding phases
    :return:
        rate: 1D tensor of the NHPP rate
    """
    #Should we use type 2 here?

    rate = finufft.nufft1d3(freqs.astype('float64'),f.astype('complex128'), (-2*np.pi*ts).astype('float64'))

    return rate

#Should I do planning?
def probe_frequencies_integral(stamps, time_total, freqs, verbose=True):
    """
    Sweeps across all the provided frequencies.
    :param time_total: exposure time in seconds
    :param freq: a tuple indicating (max_freq, window)
    :param stamps: 1D numpy containing the timestamps.
    :param verbose: whether to display estimated time of completion or not
    :return:
        f : 1D numpy array
    """
    if isinstance(time_total, int) or isinstance(time_total, float):
        stamps = stamps[stamps < time_total].astype('float64')
    elif len(time_total) == 2:
        stamps = stamps[(stamps >= time_total[0])*(stamps < time_total[1])].astype('float64') - time_total[0]
        time_total = time_total[1] - time_total[0]
    if verbose:
        tic = time.time()
    if len(freqs) == 2:
        N = int(freqs[0]/freqs[1])
        #ft_1d1 = finufft.nufft1d1(np.mod(stamps*2*np.pi*freqs[2], 2*np.pi), np.ones_like(stamps).astype('complex128')/(-2j*np.pi*stamps), 2*N+1)/time_total
        f = finufft.nufft1d1(np.mod(stamps*2*np.pi * freqs[1], 2*np.pi), np.ones_like(stamps).astype('complex128')/(-2j*np.pi*stamps), 2*N+1) / time_total
        f = f[N:]
        f = f[1:] - f[:-1]
    else:
        #Do some error stuff
        raise Exception("freqs either needs to be a 3-tuple")

    if verbose:
        print(f'Elapsed time: {time.time() - tic:03.3f} seconds')

    return f

def probe_frequencies_disjoint(stamps, time_total, left_endpoints,window_size,step_size, verbose=True):
    """
    Sweeps across all the provided frequencies.
    :param time_total: exposure time in seconds
    :param freqs: a tuple indicating (left_endpoints,window_size,step_size)
    :param stamps: 1D numpy containing the timestamps.
    :param verbose: whether to display estimated time of completion or not
    :return:
        f : 1D numpy array
    """
    if isinstance(time_total, int) or isinstance(time_total, float):
        stamps = stamps[stamps < time_total]
    elif len(time_total) == 2:
        stamps = stamps[(stamps >= time_total[0])*(stamps < time_total[1])] - time_total[0]
        time_total = time_total[1] - time_total[0]
    if verbose:
        tic = time.time()

    # set up parameters
    n_modes = 1
    n_pts = stamps.size
    nufft_type = 3
    n_trans = left_endpoints.size
    # generate nonuniform points
    x = np.mod(2 * np.pi * stamps, 2*np.pi)
    s = np.arange(0,window_size+step_size,step_size).astype('double')
    c = np.exp(2j*np.pi*left_endpoints.reshape(-1,1) *stamps.reshape(1,-1))

    plan = finufft.Plan(nufft_type, n_modes, n_trans)
    # set the nonuniform points
    plan.setpts(x,s=s)

    freqs = left_endpoints.reshape(-1,1) + s.reshape(1,-1)

    # execute the plan
    f = plan.execute(c)
    f = f /time_total
    f[freqs != 0] *2

    _, indices = np.unique(freqs, return_index = True)

    f_ab = np.zeros_like(f, dtype=np.complex128).flatten()
    f_ab[indices] = f.flatten()[indices]
    f_ab = f_ab.reshape(f.shape)

    if verbose:
        print(f'Elapsed time: {time.time() - tic:03.3f} seconds')

    return freqs, f_ab
#Now we have some bands, we need to find the frequencies fast.

def reconstruct_rate_function_disjoint(ts, f, left_endpoints,window_size,step_size, verbose = True):
    """
    Reconstructs the 1D rate function of a non-homohegeneous Poisson (NHPP) distribution.
    :param ts: 1D tensor of the time values
    :param frequencies: 1D tensor of frequencies used for the reconstruction
    :param amplitudes: 1D tensor of the corresponding amplitudes of the frequencies
    :param phases: 1D tensor of the corresponding phases
    :return:
        rate: 1D tensor of the NHPP rate
    """
    if verbose:
        tic = time.time()
    #Should we use type 2 here?
    s = np.arange(0,window_size+step_size,step_size).astype('double')

    freqs_probed = left_endpoints.reshape(-1,1) + s.reshape(1,-1)

    _, indices = np.unique(freqs_probed, return_index = True)

    f_ab = np.zeros_like(f, dtype=np.complex128).flatten()
    f_ab[indices] = f.flatten()[indices]
    f_ab = f_ab.reshape(f.shape)

    max_freq = 0.5 / np.mean(np.diff(ts))

    # set up parameters
    n_modes = 1
    n_pts = ts.size
    nufft_type = 3
    n_trans = left_endpoints[left_endpoints < max_freq].size
    x = s / (window_size) * 2 * np.pi
    s = -ts * window_size
    c = f_ab[left_endpoints < max_freq,:]

    plan = finufft.Plan(nufft_type, n_modes, n_trans)
    # set the nonuniform points
    plan.setpts(x,s=s)

    coeffs = np.exp(left_endpoints[left_endpoints < max_freq].reshape(-1,1) * ts.reshape(1,-1) * -2j*np.pi)

    rate = plan.execute(c) * coeffs

    rate = np.real(np.sum(rate, axis = 0)).flatten()

    if verbose:
        print(f'Elapsed time: {time.time() - tic:03.3f} seconds')
    return rate

#This need some tweaking
def compute_amplitude_bound_integral(stamps,time_total,window_size, percentage_in):
    """
    Computes the bound value based on the chi distribution of the estimated amplitudes.
    :param percentage_in:
    :param counts_per_sec: Normalised counts per second, e.g  timestamps.size / timestamps[-1] / exposure
    :return:
        bound_val: the estimated bound
    """
    avg_flux_per_sec = stamps.size / stamps[-1]
    y=chi.ppf(percentage_in, 2)*np.sqrt(window_size * avg_flux_per_sec / 4)/time_total
    return y

#This finds dominant frequencies in timestamp data
#Basically does exponential search on harmonics to localize and checks if its above CFAR bound
#This may have issues
def find_dominant_frequencies(timestamps, exposure_time, freq_range, search_steps = None, radius = None, multiplier = 3, maximum_frequency = 125e9, counter_threshold = 2):
    """
    This finds dominant frequencies in timestamp data
    Basically does exponential search on harmonics to localize and checks if its above CFAR bound
    This may have issues
    :param timestamps: 1D numpy containing the timestamps. Assumes that it's within the exposure
    :param time_total: exposure time in seconds
    :param freq_range: a tuple (indicating start freq, stop freq, step)
    :param search_steps: stepsize for the frequency localization
    :param multiplier: multiplier for the exponential search
    :param maximum_frequency: max frequency for the exponential search

    :param verbose: whether to display estimated time of completion or not
    :return:
        freqs_detected: the dominant frequencies, -1 when there is minimal harmonic support
    """
        
    #Probe the range of frequencies
    coeffs_fft, freqs = probe_frequencies_sweep(timestamps, exposure_time, freq_range, verbose=False, return_freqs=True)
    bound_val = compute_amplitude_bound(timestamps, exposure_time, 1 - 1/len(coeffs_fft))
    amps_all = np.abs(coeffs_fft)
    fft_ab = coeffs_fft[amps_all > bound_val]

    #Decide the search radius
    ratio = bound_val/np.max(np.abs(fft_ab))
    a = find_sinc_extrema(compute_extrema_order(ratio))/np.pi
    b = np.ceil(a)
    func = (lambda x : np.abs(np.sinc(exposure_time*x)) - ratio)
    radius_prune = bisect(func, a/exposure_time,b/exposure_time)

    if (radius_prune)/ (0.6/exposure_time) > 1:
        distance = (radius_prune)/ (0.6/exposure_time)
    else:
        distance = 1
    indices = ss.find_peaks(np.abs(np.abs(coeffs_fft)), distance=distance)[0]
    masks = np.zeros_like(freqs)
    masks [indices] = 1
    peaks = freqs[(masks * (amps_all > bound_val))> 0]

    #Issue is that there is no history, so when we do the localize_freq, we should really be make sure 
    if radius == None:
        radius = radius_prune
    if search_steps == None:
        search_steps = 0.006/exposure_time
    freqs_detected = []
    while len(peaks) > 0:
        starting_freq = peaks[0]
        peaks = peaks[1:]
        localized_freq = localize_freq(timestamps, exposure_time, starting_freq, radius, search_steps, multiplier, maximum_frequency, counter_threshold)
        
        doAppend = True
        #removes detected frequencies that are harmonics of previously detected frequencies.
        for dom_freq in freqs_detected:
            difference = np.minimum(np.mod(localized_freq,dom_freq), dom_freq - np.mod(localized_freq,dom_freq))
            if difference < radius:
                doAppend = False
                break
        if doAppend:
            freqs_detected.append(localized_freq)
        differences = np.minimum(np.mod(peaks,localized_freq), localized_freq - np.mod(peaks,localized_freq))
        peaks = peaks[differences > radius]
        
    return np.unique(np.array(freqs_detected))
    
#Approximate the location of nth extrema of a sinc function
def find_sinc_extrema(n = 1):
    q = np.pi * (n+0.5)
    #https://math.stackexchange.com/questions/4180499/infinite-series-for-the-x-coordinate-of-the-n-textth-peak-valley-of-t
    return q - 1/q - 2/3*q**-3- 13/15*q**-5- 146/105*q**-7 - 781/315*q**-9-378193/80640*q**-11 - 1043207/120960*q**-13

#Computes the largest extrema location for which the bound can be less than
def compute_extrema_order(bound):
    counter = 0
    while np.abs(np.sinc(find_sinc_extrema(counter)/(np.pi))) > bound:
        counter = counter + 1
    return counter - 1


#localizes a frequency given a timestamp by probing harmonics
def localize_freq(stamps, time_total, starting_freq, radius, steps, multiplier = 3, max_frequency = 125e9, counter_threshold = 2):
    central_freq = starting_freq
    counter = 0
    while central_freq < max_frequency:
        fft,freqs = probe_frequencies_sweep(stamps, time_total, (central_freq - radius, central_freq + radius,steps), verbose=False, return_freqs=True)
        percentageIn = 1 - 1/ (freqs.size)
        bound_val_new = compute_amplitude_bound(stamps, time_total, percentageIn)
        if np.sum(np.abs(fft) > bound_val_new) == 0:
            break
        central_freq = freqs[np.argmax(np.abs(fft))]*multiplier
        counter += 1

    if counter < counter_threshold:
        return -1
    else:
        return central_freq / (multiplier**counter)

def apply_deadtime(stamps):
    stamps = np.sort(stamps, kind='mergesort').tolist()
    prev = 0
    new_stamps = []
    for stamp in stamps:
        if stamp - prev >= 2.31e-7:
            new_stamps.append(stamp)
            prev = stamp
    return np.array(new_stamps)

def gaussian_pulse(amplitude, std, center, x):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * std ** 2))

def create_gaussian_pulse(amplitude, std, center, size):
    x = np.arange(size)
    return gaussian_pulse(amplitude, std, center, x)

def func1(num_photons, shape, exposure, device='cuda:0'):
    torch.manual_seed(7252)
    u1 = np.random.rand(shape[0], int(int(1e6) * num_photons * exposure))
    u1 = u1[u1 != 0].reshape((1, -1))
    # print("u1 is done")
    s = np.cumsum(-np.log(u1), axis=1)
    del u1
    return s

def get_timestamps(flux, exposure=1, num_photons=1000, device='cuda:0'):
    s = func1(num_photons, (1, flux.shape[0]), exposure, device=device)
    mean_val = np.mean(flux)
    rate_func = (flux / mean_val * exposure * 1e4)
    rate_func[np.isnan(rate_func)] = 0
    max_val = np.max(rate_func)
    t_act = s[0] / max_val
    t_act = t_act[t_act < exposure]
    t_new = np.floor(t_act * flux.shape[0]).astype('int')
    u2 = np.random.rand(t_act.shape[0])
    t_accepted = []
    # u2 = u2.cpu().numpy().tolist()
    t_new = t_new.tolist()
    for t in tqdm.tqdm(range(len(t_new))):
        result_value = flux[t_new[t]]
        if u2[t] <= result_value / max_val:
            t_accepted.append(t_act[t].item())
    return t_accepted

def timestamp_simulation_pulse(rep_rate, time_total, device='cuda:0', sampling_res=1e14, pulse_fwhm=2e-10, amplitude=1e7, num_photons=1000):
    array_size = int(2 * sampling_res / 1e9)
    ones_loc = int(sampling_res / 1e9)
    sigma = int(pulse_fwhm * sampling_res)
    pulse = create_gaussian_pulse(amplitude, sigma, ones_loc, array_size)
    t_accepted = get_timestamps(pulse, num_photons=num_photons, device=device)
    centers = np.arange(0, int(1e9), step=int(1e9 / rep_rate))
    shifts = np.random.choice(centers, size=len(t_accepted))
    # Converts to nano-second
    t_new = np.array(t_accepted) / (sampling_res / 1e9)
    t_new += shifts
    t_new /= 1e9
    t_new = apply_deadtime(t_new)
    t_accepted = t_new[t_new < time_total]
    return t_accepted