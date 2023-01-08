import numpy as np 
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import mne
import scipy
from scipy.signal import *
from scipy.stats import gamma
import scipy.special as sc

def reorder(ori, shuffled):
    return np.where(shuffled.reshape(-1, 1) == ori.reshape(1, -1))[1]


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def read_raw(raw_path):
    """Read raw data from an EDF file
    
    Parameters
    ----------
    raw_path : str
        Path to the EDF file
    
    Returns
    -------
    data : numpy array
    channels : numpy array"""
    raw_path = validate_type(raw_path, 'raw_path', str)
    raw = mne.io.read_raw_edf(raw_path)
    raw_channels = raw.info['ch_names']
    data = []

    for raw_ch in raw_channels:
        ch_data = raw.get_data(raw_ch) * 1E6
        data.append(ch_data)

    data = np.squeeze(data)
    channels = np.array(raw_channels)
    return data, channels


def preprocess(raw, sample_freq, filter_freq, rp=0.5, rs=93, space=0.5):
    """Preprocess the raw data
    Parameters
    ----------
    raw : numpy array like of float, shape (n_samples,)
        raw data
    sample_freq : float
        sample frequency
    filter_freq : list of float, shape (2,)
        filter frequency
    rp : float | 0.5
        max loss in passband (dB)
    rs : float | 93
        min attenuation in stopband (dB)
    space : float

    Returns
    -------
    filtered : array
        filtered data
    """
    # check raw data
    raw = validate_type(raw, 'raw', np.ndarray)
    if raw.ndim != 1:
        raise ValueError('raw must be a 1D array')
    # check sample frequency
    sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
    # check filter frequency
    filter_freq = validate_filter_freq(filter_freq)

    nyq = sample_freq / 2
    # MGNM
    if filter_freq[1] >= .99 * nyq:
        filter_freq[1] = nyq * .99
    low, high = filter_freq

    scale = 0
    while 0 < low < 1:
        low *= 10
        scale += 1
    low = filter_freq[0] - (space * 10 ** (-1 * scale))

    scale = 0
    while high < 1:
        high *= 10
        scale += 1
    high = filter_freq[1] + (space * 10 ** (-1 * scale))  
    stop_freq = [low, high]
    ps_order, wst = cheb2ord([filter_freq[0] / nyq, filter_freq[1] / nyq], [stop_freq[0] / nyq, stop_freq[1] / nyq], rp,
                                rs)
    z, p, k = cheby2(ps_order, rs, wst, btype='bandpass', analog=0, output='zpk')
    sos = zpk2sos(z, p, k)    

    filtered = sosfilt(sos, raw)
    filtered = sosfilt(sos, np.flipud(filtered))
    filtered = np.flipud(filtered)

    return filtered


def validate_type(var, var_name, var_type):
    """Validate the type of a variable"""
    if not isinstance(var, var_type):
        raise TypeError('{} must be a {}'.format(var_name, var_type))
    return var


def validate_param(var, var_name, var_type):
    """Validate the type and value of a variable"""
    var = validate_type(var, var_name, var_type)
    if var < 0:
        raise ValueError('{} must not be a negative {}'.format(var_name, var_type))
    return var


def validate_filter_freq(filter_freq):
    """Validate filter frequency
    Parameters
    ----------
    filter_freq : array
        filter frequency
    Returns
    -------
    filter_freq : array
        filter frequency
    """
    filter_freq = validate_type(filter_freq, 'filter_freq', (list, tuple, np.ndarray))
    if len(filter_freq) != 2:
        raise ValueError('filter_freq must have two elements')
    if not all(isinstance(f, (int, float)) for f in filter_freq):
        raise TypeError('filter_freq must be a list of int or float')   
    if filter_freq[0] >= filter_freq[1]:
        raise ValueError('filter_freq[0] must be smaller than filter_freq[1]')
    if filter_freq[0] <= 0:
        raise ValueError('filter_freq[0] must be positive')
    return filter_freq


def compute_rms(filtered, sample_freq, window_size=6*1e-3, detector=None):
    """Compute RMS
    Parameters
    ----------
    filtered : numpy array like of float, shape (n_samples,)
        filtered data
    sample_freq : float | int
        sample frequency
    window_size : float | int
        window size
    detector : str
        detector type
    Returns
    -------
    rms : array of float, shape (n_samples,)
        RMS
    """
    # check filtered data
    filtered = validate_type(filtered, 'filtered', np.ndarray)
    if filtered.ndim != 1:
        raise ValueError('filtered must be a 1D array')
    # check sample frequency
    sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
    # check window size 
    window_size = validate_param(window_size, 'window_size', (int, float))
    # check detector
    if detector is not None:
        detector = validate_type(detector, 'detector', str)
        if detector not in ['RMS', 'MNI']:
            raise ValueError('detector must be RMS or MNI')

    # compute RMS
    if detector == 'RMS':
        rms_window = round(window_size * sample_freq)
    elif detector == 'MNI':
        rms_window = round(0.002 * sample_freq)
    
    if rms_window % 2 == 0:
        rms_window += 1
    
    temp = np.square(filtered)
    temp = lfilter(np.ones(rms_window), 1, temp, axis=0) / rms_window
    rms = np.zeros(len(temp))
    rms[:int(-np.floor(rms_window / 2))] = temp[int(np.floor(rms_window / 2)):]
    rms = np.sqrt(rms)

    return rms


def compute_wavelet_entropy(sample_freq, epoch_samples, low, high, dev_cycles=3):
    """Compute wavelet entropy
    Parameters
    ----------
    sample_freq : int | float
        sample frequency
    epoch_samples : int | float
        epoch samples
    low : int | float
        low frequency
    high : int | float
        high frequency
    dev_cycles : int | float | 3 | optional
        number of cycles
    Returns
    -------
    wavelet_entropy : int | float
        wavelet entropy"""
    #check inputs
    sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
    epoch_samples = validate_param(epoch_samples, 'epoch_samples', (int, float))
    low = validate_param(low, 'low', (int, float))
    high = validate_param(high, 'high', (int, float))
    dev_cycles = validate_param(dev_cycles, 'dev_cycles', (int, float))
    
    wavelet_entropy_max = np.zeros((100,1))
    for i in range(0, 100):
        segment = np.random.rand(epoch_samples, 1)
        auto_corr = correlate(segment, segment, mode='full') / np.sum(segment ** 2)
        w_coef = gabor_transform_wait(auto_corr, sample_freq, low, high, dev_cycles)
        prob_energy = np.mean(w_coef ** 2, 1) / np.sum(np.mean(w_coef ** 2, 1))
        wavelet_entropy_max[i] = -np.sum(prob_energy * np.log(prob_energy))

    del segment, prob_energy, w_coef

    return np.median(wavelet_entropy_max)


def gabor_transform_wait(pv_signal, sample_rate, min_freq, max_freq, dev_cycles=3, magnitudes=1,
                         squared_mag=0,
                         band_ave=0, phases=0, time_step=[]):
    """This function calculates the Wavelet Transform using a Gaussian modulated window (Gabor Wavelet).
    Parameters
    ----------
    pv_signal : array
        Signal to process
    sample_rate : float | int
        sample rate
    min_freq : float | int
        Min frequency (in Hz) to process from
    max_freq : float | int
        Max frequency (in Hz) to process to
    dev_cycles : float | int
         the number of cycles to include for the wavelet transform at every frequency.
    magnitudes : int|float | 1 | optional
        Set to 1 (default) if the magnitudes of the coefficients must be returned; 0 for analytic values (complex values).
    squared_mag : int|float | 0 | optional
        Set to 1 if the magnitudes of the coefficients divided by the squared of the corresponding scale must by power to 2
    band_ave : int|float | 0 | optional
        Set to 1 if instead of returning a matrix with all values
        in the time-frequency map, the function returns just a vector with the
        average along all the frequency scales for each time moment.
    phases : int | 0 | optional
        Set to 1 if the phases of the coefficients must be returned; 0 for analytic values (complex values).
    time_step : array|list | [] | optional
        time step between values that are going to be kept in the output matrix
        Each time moment is the average of the previous values according to the size of the window defined by this parameter.
    
    Returns
    -------
    w_coef : array
        wavelet transform output matrix (n_freqs x n_samples) Time in rows, Frequency in columns
    """
    # check inputs
    pv_signal = validate_type(pv_signal, 'pv_signal', np.ndarray)
    sample_rate = validate_param(sample_rate, 'sample_rate', (int, float))
    min_freq = validate_param(min_freq, 'min_freq', (int, float))
    max_freq = validate_param(max_freq, 'max_freq', (int, float))
    dev_cycles = validate_param(dev_cycles, 'dev_cycles', (int, float))
    magnitudes = validate_param(magnitudes, 'magnitudes', (int, float))
    squared_mag = validate_param(squared_mag, 'squared_mag', (int, float))
    band_ave = validate_param(band_ave, 'band_ave', (int, float))
    phases = validate_param(phases, 'phases', (int, float))
    time_step = validate_type(time_step, 'time_step', (list, tuple, np.ndarray))

    pv_signal = pv_signal[:]
    freq_seg = len(list(range(min_freq, max_freq, 5)))
    freq_step = int((max_freq - min_freq) / (freq_seg - 1))
    freq_axis = np.arange(min_freq, max_freq + freq_step, freq_step)
    freq_axis = np.flip(freq_axis)
    if len(pv_signal) % 2 == 0:
        pv_signal = pv_signal[0:len(pv_signal) - 1]
    
    time_axis = (np.arange(0, len(pv_signal))) / sample_rate
    s_len = len(time_axis)
    s_half_len = (s_len // 2) + 1

    w_axis = (2 * np.pi / s_len) * (np.arange(0, s_len))
    w_axis = w_axis * sample_rate
    w_axis_half = w_axis[:s_half_len]

    if not time_step:
        sample_ave = 1
    else:
        sample_ave = [round(ts * sample_rate) for ts in time_step]
        if all(x < 1 for x in sample_ave):
            sample_ave = 1

    sample_ave_filt = []
    if sample_ave > 1:
        ind_samp = list(range(1, len(time_axis), sample_ave))
        time_axis = [time_axis[i] for i in ind_samp]
        sample_ave_filt = np.ones(sample_ave, 1)

    signal_fft = scipy.fft.fft(pv_signal, len(pv_signal), axis=0)
    signal_fft = signal_fft.reshape((-1, 1))

    gabor_wt = np.zeros((len(freq_axis), len(time_axis)), dtype=complex)
    freq_ind = 0
    for freq_counter in freq_axis:
        dev_sec = (1 / freq_counter) * dev_cycles 
        win_fft = np.zeros((s_len, 1))
        tmp1 = (w_axis_half - 2 * np.pi * freq_counter) ** 2
        tmp2 = dev_sec ** 2
        tmp = np.exp(-0.5 * tmp1 * tmp2)
        win_fft[:s_half_len] = np.reshape(tmp, (len(tmp), 1))
        win_fft = win_fft * np.sqrt(s_len) / np.linalg.norm(win_fft)
        freq_ind += 1

        if sample_ave > 1:
            gabor_tmp = np.zeros((len(signal_fft) + (sample_ave - 1), 1), dtype=complex)
            gabor_tmp[sample_ave:] = scipy.fft.ifft(signal_fft * win_fft, axis=0) / np.sqrt(dev_sec)

            if magnitudes:
                gabor_tmp = abs(gabor_tmp)
            if squared_mag:
                gabor_tmp **= 2

            gabor_tmp[:(sample_ave - 1)] = np.flipud(gabor_tmp[(sample_ave + 1):(2 * sample_ave - 1)])
            gabor_tmp = filter(sample_ave_filt, 1, gabor_tmp) / sample_ave
            gabor_tmp = gabor_tmp[sample_ave:]

            gabor_wt[freq_ind - 1] = gabor_tmp[ind_samp]

        else:
            tmp_ifft = scipy.fft.ifft(signal_fft * win_fft, axis=0) / np.sqrt(dev_sec)
            gabor_wt[freq_ind - 1] = tmp_ifft.reshape((1, len(tmp_ifft)))

    if sample_ave > 1:
        return gabor_wt

    if phases != 0:
        gabor_wt = np.angle(gabor_wt)
        return gabor_wt

    if magnitudes != 1:
        return gabor_wt

    gabor_wt = abs(gabor_wt)

    if squared_mag != 0:
        gabor_wt **= 2

    if band_ave != 0:
        gabor_wt = np.mean(gabor_wt, 2)
        gabor_wt = np.flipud(gabor_wt)
        time_axis = []
        freq_axis = np.fliplr(freq_axis)

    return gabor_wt


def pos_baseline(baseline_window, rms, epoch_time, sample_freq, thr_perc):
    """Calculate baseline for positive peaks.

    Parameters
    ----------
    baseline_window : array
        baseline window
    rms : array
        rms
    epoch_time : int|float
        epoch time
    sample_freq : int|float
        sample frequency
    thr_perc : int|float
        threshold percentage
    Returns
    ------- 
    res : array
        baseline
    """
    # check inputs
    baseline_window = validate_type(baseline_window, 'baseline_window', np.ndarray)
    if baseline_window.ndim != 2 and baseline_window.shape[1] != 1:
        raise ValueError('baseline_window must be a 2D array with shape (n_samples, 1)')
    if not np.equal(baseline_window, 0).any() or not np.equal(baseline_window, 1).any():
        raise ValueError('baseline_window contains only 0 and 1')
    rms = validate_type(rms, 'rms', np.ndarray)
    epoch_time = validate_param(epoch_time, 'epoch_time', (int, float))
    sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
    thr_perc = validate_param(thr_perc, 'thr_perc', (int, float))

    # calculate baseline
    window_thrd = round(epoch_time * sample_freq)
    bw_tmp = np.zeros((len(baseline_window)+2, 1))
    bw_tmp[1:-1] = baseline_window
    win_baseline = np.diff(bw_tmp, axis=0)
    win_baseline_up = np.where(win_baseline == 1)[0] + 1
    win_baseline_down = np.where(win_baseline == -1)[0]
    
    idx_ini = []
    for i in range(len(win_baseline_up)):
        idx_add = list(range(win_baseline_up[i], win_baseline_down[i], window_thrd))
        idx_ini.extend(idx_add)
    idx_ini = np.array(idx_ini)
    idx_end = idx_ini + window_thrd - 1
    idx_rem = idx_end <= len(rms)
    idx_ini = idx_ini[idx_rem]
    idx_end = idx_end[idx_rem]       

    res = np.zeros(rms.shape)
    for i in range(len(idx_ini)):
        sect = np.sort(rms[int(idx_ini[i]) - 1:int(idx_end[i])])
        gam_params = fb_gamfit(sect)
        percent = fb_gamcdf(sect, gam_params[0], gam_params[1])
        p_idxs = np.where(percent <= thr_perc)
        pi = p_idxs[0][-1]
        res[int(idx_ini[i]) - 1:int(idx_end[i])] = sect[pi]

    return res


def neg_baseline(rms, min_win, sample_freq, epo_CHF, per_CHF):
    """Calculate baseline for negative peaks.

    Parameters
    ----------
    rms : array
    min_win : int
        minimum HFO Time window
    sample_freq : int|float 
        sample frequency
    epo_CHF : int|float
        Continuous High Frequency Epoch
    per_CHF : int|float
        Continuous High Frequency Percentile Threshold
    
    Returns
    -------
    res : array
        baseline
    """
    # check inputs
    rms = validate_type(rms, 'rms', np.ndarray)
    min_win = validate_param(min_win, 'min_win', (int, float))
    sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
    epo_CHF = validate_param(epo_CHF, 'epo_CHF', (int, float))
    per_CHF = validate_param(per_CHF, 'per_CHF', (int, float))

    # calculate baseline
    min_win = min_win * sample_freq
    window_chf = round(epo_CHF * sample_freq)
    idx_ini = np.arange(0, len(rms), window_chf)
    idx_end = idx_ini + window_chf - 1
    idx_rem = idx_end <= len(rms)
    idx_ini = idx_ini[idx_rem]
    idx_end = idx_end[idx_rem]

    res = np.zeros(rms.shape)
    for i in range(len(idx_ini)):
        sect = np.sort(rms[int(idx_ini[i]):int(idx_end[i]) + 1], axis=0)
        thres_last = np.max(sect)
        while True:
            if sum(abs(sect)) == 0:
                break
            # MLE of params of gamma distribution fit to the data(sect), return gam_params = (shape, scale)
            if np.all(sect != 0):
                gam_params = gamma.fit(sect, floc=0)
            else:
                gam_params = gamma_fit(sect)
            percent = gamma.cdf(sect, gam_params[0], gam_params[1], gam_params[2])
            p_idxs = np.where(percent <= per_CHF)
            thres_new = sect[p_idxs[0][-1]]
            energy_over = np.zeros(len(sect)+2)
            energy_over[1:-1] = sect >= thres_new
            jumps = np.diff(energy_over, axis=0)
            jump_up = np.where(jumps == 1)[0]
            jump_down = np.where(jumps == -1)[0] - 1
            dist = jump_down - jump_up

            select = np.where(dist > min_win)[0]
            if select.size == 0:
                break
            ini = jump_up[select]
            end = jump_down[select]

            for ii in range(len(ini)):
                sect[ini[ii]: end[ii] + 1] = 0
                sect = np.sort(sect,axis=0)

            thres_last = thres_new

        res[int(idx_ini[i]):int(idx_end[i]) + 1] = thres_last

    return res


def gampdf(x, a, b):
    """Gamma probability density function.

    Parameters
    ----------
    x : array
        input values
    a : int|float
        parameter of the gamma distribution
    b : int|float
        parameter of the gamma distribution

    Returns
    -------
    res : array
        the probability density function evaluated at x
    """
    # check inputs
    x = validate_type(x, 'x', np.ndarray)
    a = validate_type(a, 'a', (int, float))
    b = validate_type(b, 'b', (int, float))
    # calculate
    sz = len(x)
    pdf = np.zeros(sz)

    k = np.where(np.logical_or(a <= 0, b <= 0, np.isnan(x)))[0]
    if k.size != 0:
        pdf[k] = np.nan

    k = np.where(np.logical_and(x > 0, 0 < a <= 1 and b > 0))[0]
    if k.size != 0:
        if np.isscalar(a) and np.isscalar(b):
            pdf[k] = (x[k] ** (a-1)) * np.exp(-x[k] / b) / sc.gamma(a) / (b ** a)
        else:
            pdf[k] = (x[k] ** (a[k]-1)) * np.exp(-x[k] / b[k]) / sc.gamma(a[k]) / (b[k] ** a[k])

    k = np.where(np.logical_and(x > 0, a > 1 and b > 0))[0]
    if k.size != 0:
        if np.isscalar(a) and np.isscalar(b):
            pdf[k] = np.exp(-a * np.log(b) + (a-1) * np.log(x[k]) - x[k] / b - sc.gammaln(a))
        else:
            pdf[k] = np.exp(-a[k] * np.log(b[k]) + (a[k]-1) * np.log(x[k]) - x[k] / b[k] - sc.gammaln(a[k]))

    return pdf

            
def gamlike(p, x):
    """Negative log-likelihood function for the Gamma distribution over vector x, with the given parameters p[0] and p[1].

    Parameters
    ----------
    p : array
        parameters of the gamma distribution (shape, scale)
    x : array
        input values

    Returns
    -------
    res : array
        the negative log-likelihood function evaluated at x with the given parameters
    """
    # check inputs
    p = validate_type(p, 'p', (list, np.ndarray))
    if len(p) != 2:
        raise ValueError('p must be a list or array with 2 elements')
    p[0] = validate_type(p[0], 'p[0]', (int, float))
    p[1] = validate_type(p[1], 'p[1]', (int, float))
    x = validate_type(x, 'x', np.ndarray)

    return -np.sum(np.log(gampdf(x, p[0], p[1])))


def gamfit_search(a, avg, x):
    """Search for the best fit of gamma distribution.

    Parameters
    ----------
    a : int|float
    avg : int|float
    x : array

    Returns
    -------
    res : array
    """
    # check inputs
    a = validate_type(a, 'a', (int, float))
    avg = validate_type(avg, 'avg', (int, float))
    x = validate_type(x, 'x', np.ndarray)  
    # calculate
    return -gamlike([a, avg / a], x)


def nmsmax(func, x, stopit=[], savit=[], *args):
    """Nelder-Mead simplex method for direct search optimization.

    Parameters
    ----------
    func : function
        function to be optimized
    x : list|np.ndarray 
        initial guess
    stopit : list|np.ndarray 
        stopping criteria
    saveit : list|np.ndarray 
        saving criteria
    args : array|tuple|list
        additional arguments for the function

    Returns
    -------
    x[0] : array
        vector yielding largest function value found
    fmax : float   
        function value at x
    nf : int
        number of function evaluations
    """
    x = validate_type(x, 'x', (list, np.ndarray))
    stopit = validate_type(stopit, 'stopit', (list, np.ndarray))
    savit = validate_type(savit, 'savit', (list, np.ndarray))

    x0 = x[:]
    n = len(x0)

    stopit = [1e-3, np.inf, np.inf, 0, 0]
    tol, trace = stopit[0], stopit[4]

    V = np.hstack((np.zeros((n, 1)), np.eye(n)))
    f = np.zeros((n + 1, 1))
    V[:, 0] = x0
    f[0] = func(1, *args)
    fmax_old = f[0]

    k, m = 0, 0

    scale = np.max((np.linalg.norm(np.array(x), np.inf), 1))
    alpha = scale / (n * np.sqrt(2)) * np.array([np.sqrt(n + 1) - 1 + n, np.sqrt(n + 1) - 1])
    V[:, 1:n + 1] = x0 + alpha[1] * np.ones((n, 1)) * np.ones((1, n))
    for i in range(1, n + 1):
        V[i - 1, i] = x0[i - 1] + alpha[0]
        x[:] = V[:, i]
        f[i] = func(x[0], *args)
    nf = n + 1
    temp = np.sort(f)
    idx = np.where(temp)[0]
    idx = idx[::-1]
    f = f[idx]
    V = V[:, idx]

    alpha, beta, gamma = 1, 1 / 2, 2

    while True:
        k += 1
        fmax = f[0]
        if fmax > fmax_old:
            if len(savit) != 0:
                x[:] = V[:, 0]
        fmax_old = fmax
        # three stopping tests
        # Stopping Test 1 - f reached target value
        if fmax >= stopit[2]:
            break
        # Stopping Test 2 - too many f-evals
        if nf >= stopit[1]:
            break
        # Stopping Test 3 - converged
        v1 = V[:, 0]
        size_simplex = np.linalg.norm(V[:, 1:n + 1] - np.tile(v1, n), 1) / np.max((1, np.linalg.norm(v1, 1)))
        if size_simplex <= tol:
            break

        # One step of the Nelder-Mead simplex algorithm
        vbar = np.transpose(np.sum(np.transpose(V[:, 0:n]) / n))
        vr = (1 + alpha) * vbar - alpha * V[:, n]
        x[:] = vr
        fr = func(x[0], *args)
        nf += 1
        vk, fk = vr, fr
        if fr > f[n - 1]:
            if fr > f[0]:
                ve = gamma * vr + (1 - gamma) * vbar
                x[:] = ve
                fe = func(x[0], *args)
                nf += 1
                if fe > f[0]:
                    vk, fk = ve, fe
        else:
            vt = V[:, n]
            ft = f[n]
            if fr > ft:
                vt, ft = vr, fr
            vc = beta * vt + (1 - beta) * vbar
            x[:] = vc
            fc = func(x[0], *args)
            nf += 1
            if fc > f[n - 1]:
                vk, fk = vc, fc
            else:
                for j in range(1, n):
                    V[:, j] = (V[:, 0] + V[:, j]) / 2
                    x[:] = V[:, j]
                    f[j] = func(x[0], *args)
                nf += (n - 1)
                vk = (V[:, 0] + V[:, n]) / 2
                x[:] = vk
                fk = func(x[0], *args)
                nf += 1
        V[:, n] = vk
        f[n] = fk
        temp = np.sort(f)
        idx = np.where(temp)[0]
        idx = idx[::-1]
        f = f[idx]
        V = V[:, idx]

    x[:] = V[:, 0]
    return x[0], fmax[0], nf


def fb_gamfit(x):
    """the maximumlikelihood estimator for the Gamma distribution for x

    Parameters
    ----------
    x : array
        data

    Returns
    ------- 
    res : list of float, length 2
        [a, b] where a is the shape and b is the scale
    """
    # check inputs
    x = validate_type(x, 'x', np.ndarray)
    # fit gamma distribution
    avg = np.mean(x)
    a, _, _ = nmsmax(gamfit_search, [1], [], [], avg, x)
    b = a / avg
    res = [a, 1 / b]
    return res


def fb_gamcdf(x, a, b):
    """the cumulative distribution function for the Gamma distribution for x given a and b

    Parameters
    ----------
    x : array
        data
    a : float | int
        shape
    b : float | int
        scale

    Returns
    -------
    cdf : array
        cumulative distribution function values for x 
    """

    # check inputs
    x = validate_type(x, 'x', np.ndarray)
    a = validate_type(a, 'a', (int, float))
    b = validate_type(b, 'b', (int, float))

    # calculate cdf
    sz = len(x)
    cdf = np.zeros(sz)

    k = np.where(np.logical_or(a <= 0, b <= 0, np.isnan(x)))[0]
    if k.size != 0:
        cdf[k] = np.nan

    k = np.where(np.logical_and(x > 0, a > 0 and b > 0))[0]
    if k.size != 0:
        if np.isscalar(a) and np.isscalar(b):
            cdf[k] = sc.gammainc(a, x[k]/b)
        else:
            cdf[k] = sc.gammainc(a[k], x[k]/b[k])
    return cdf


def gamma_fit(x):
    n = len(x)
    freq = 1
    scalex = np.sum(freq*x) / n
    x = x / scalex
    xbar = 1
    s2 = np.sum(freq*(x-xbar)**2) / n
    s2 = s2 * n / (n - 1)
    ahat = xbar ** 2 / s2
    bhat = s2 / xbar
    parmhat = [ahat, 0, bhat*scalex]
    # parmhat[2] = parmhat[2] * scalex
    return parmhat





    
