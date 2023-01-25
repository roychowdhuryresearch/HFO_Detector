import numpy as np
from scipy.signal import *
from .utils import *

class MNIDetector():

    def __init__(self, sample_freq, filter_freq=[80, 500], 
                epoch_time=10, epo_CHF=60, per_CHF=95/100, 
                min_win=10*1e-3, min_gap=10*1e-3, thrd_perc=99.9999/100, 
                base_seg=125*1e-3, base_shift=0.5, base_thrd=0.67, base_min=5,
                n_jobs=32, use_kwargs=True, front_num=1, seed=None):
        """Initialize the MNI detector
        sample_freq : float | int
            Sampling frequency of the data in Hz
        filter_freq : list of float or int, length 2, default [80, 500] Hz
            Filter freqs in Hz
        epoch_time: float | int, default 10 s
            Cycle Time in seconds
        epo_CHF: float | int, default 60 s
            Continuous High Frequency epoch in seconds
        per_CHF: float | int, default 95 %
            Continous High Frequency percentage (%)
        min_win: float | int, default 10 ms
            Minimum HFO time in seconds
        min_gap: float | int, default 10 ms
            Minimum gap between HFOs in seconds
        thrd_perc: float | int, default 99.9999 %
            Threshold percentile (%)
        base_seg: float | int, default 125 ms
            Baseline window size in seconds
        base_shift: float | int, default 0.5
            Baseline window shift
        base_thrd: float | int, default 0.67
            Baseline threshold 
        base_min: float | int, default 5
            Baseline minimum time
        n_jobs : int, default 32
            Number of jobs to run in parallel
        use_kwargs : bool, default True
            Whether to pass additional arguments to the map function
        front_num : int, default 1
            Number of jobs to run in parallel at the front end
        seed : int, default None
            Random seed for reproducibility
        """
        sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
        filter_freq = validate_filter_freq(filter_freq)
        if filter_freq[1] > sample_freq/2:
            raise ValueError('filter_freq[1] must be less than sample_freq/2')
        epoch_time = validate_param(epoch_time, 'epoch_time', (int, float))
        epo_CHF = validate_param(epo_CHF, 'epo_CHF', (int, float))
        per_CHF = validate_param(per_CHF, 'per_CHF', (int, float))
        min_win = validate_param(min_win, 'min_win', (int, float))
        min_gap = validate_param(min_gap, 'min_gap', (int, float))
        thrd_perc = validate_param(thrd_perc, 'thrd_perc', (int, float))       
        base_seg = validate_param(base_seg, 'base_seg', (int, float))
        base_shift = validate_param(base_shift, 'base_shift', (int, float))
        base_thrd = validate_param(base_thrd, 'base_thr', (int, float))
        base_min = validate_param(base_min, 'base_min', (int, float))
        n_jobs = validate_param(n_jobs, 'n_jobs', int)
        use_kwargs = validate_param(use_kwargs, 'use_kwargs', bool)
        front_num = validate_param(front_num, 'front_num', int)
        seed = validate_param(seed, 'seed', (int, type(None)))


        self.sample_freq = sample_freq
        self.filter_freq = filter_freq
        self.epoch_time = epoch_time
        self.epo_CHF = epo_CHF
        self.per_CHF = per_CHF
        self.min_win = min_win
        self.min_gap = min_gap
        self.thrd_perc = thrd_perc
        self.base_seg = base_seg
        self.base_shift = base_shift
        self.base_thrd = base_thrd
        self.base_min = base_min
        self.n_jobs = n_jobs
        self.use_kwargs = use_kwargs
        self.front_num = front_num   
        self.seed = seed 

    def detect_edf(self, edf_path):
        """Detect HFOs from an EDF file
        
        Parameters
        ----------
        edf_path : str
            Path to the EDF file
            
        Returns
        -------
        channel_names : numpy array of str, shape (n_channels,)
        HFOs : numpy array of int, shape (n_channels, n_HFOs, 2)"""
        edf_path = validate_type(edf_path, 'edf_path', str)
        raw, channels = read_raw(edf_path)
        return self.detect_multi_channels(data=raw, channels=channels) 


    def detect_multi_channels(self, data, channels, filtered=False):
        """Detect HFOs from numpy arrays of multi channels parrallelly
        
        Parameters
        ----------
        data : numpy array like of float, shape (n_channels, n_samples)
        channels : numpy array like of str, shape (n_channels,)
        filtered : bool, default False, 
            Whether the data is already filtered

        Returns
        -------
        channel_names : numpy array of str, shape (n_channels,)
        HFOs : numpy array of int, shape (n_channels, n_HFOs, 2)"""
        
        data = validate_type(data, 'data', np.ndarray)
        channels = validate_type(channels, 'channels', np.ndarray)

        param_list = [{"data":data[i], "channel_names":channels[i], "filtered":filtered} for i in range(len(channels))]
        ret = parallel_process(param_list, self.detect, self.n_jobs, self.use_kwargs, self.front_num)
        channel_name, HFOs = [], []
        for j in ret:
            if not type(j) is tuple:
                print(j)
            if j[0] is None:
                continue
            HFOs.append(j[0])
            channel_name.append(j[1])
        channel_name = np.array(channel_name)
        HFOs = np.array(HFOs, dtype=object)
        index = reorder(channel_name, channels)
        return channel_name[index], HFOs[index]


    def detect(self, data, channel_names, filtered=False):
        """Detect HFOs from a single channel

        Parameters
        ----------
        data : numpy array like of float, shape (n_samples,)
        channel_names : str

        Returns
        -------
        HFOs : numpy array of int, shape (n_HFOs, 2)
        channel_names : str"""

        data = validate_type(data, 'data', np.ndarray)
        if data.shape[0] == 0:
            return None, None
        if data.ndim != 1:
            raise ValueError('data must be a 1D array')
        channel_names = validate_type(channel_names, 'channel_names', str)
        if not filtered:
            data = preprocess(data, self.sample_freq, self.filter_freq)
        rms = compute_rms(data, self.sample_freq, detector='MNI')
        baseline_window = self._detect_baseline(data, self.sample_freq, self.filter_freq, self.base_seg, self.base_shift, self.base_thrd, self.seed)
        thrd = self._compute_thrd(len(data), rms, baseline_window, self.base_min, self.epoch_time, self.sample_freq, self.thrd_perc, self.min_win, self.epo_CHF, self.per_CHF)
        HFOs = self._get_HFOs(rms, thrd, self.min_win, self.sample_freq, self.min_gap)
        return HFOs, channel_names


    def _detect_baseline(self, filtered, sample_freq, filter_freq, base_seg, base_shift, base_thrd, seed):
        """Detect baseline segments
        
        Parameters
        ----------
        filtered : numpy array of float, shape (n_samples,)
        sample_freq : int or float
        filter_freq : list
        base_seg : float or int
        base_shift : float or int  
        base_thr : float or int
        
        Returns
        -------
        baseline_window : numpy array"""

        # check inputs
        if filtered.ndim != 1:
            raise ValueError('filtered must be a 1D array')

        # init outputs
        data_len = len(filtered)
        baseline_window = np.zeros((data_len, 1))

        # get wavelet entropy
        epoch_samples = round(sample_freq * base_seg)
        low, high = filter_freq
        wavelet_entropy = compute_wavelet_entropy(sample_freq, epoch_samples, low, high, seed)
        
        # get baseline
        in_idx = list(range(1, data_len + 1, round(epoch_samples * base_shift)))
        en_idx = [idx + epoch_samples - 1 for idx in in_idx]
        idx = [1 if i > data_len else 0 for i in en_idx]
        in_idx = in_idx[: idx.index(1)]
        en_idx = en_idx[: idx.index(1)]
        del idx

        for i in range(0, len(in_idx)):
            sig_filt = filtered[in_idx[i] - 1:en_idx[i]]
            auto_corr = correlate(sig_filt, sig_filt) / sum(sig_filt ** 2)
            w_coef = gabor_transform_wait(auto_corr, sample_freq, low, high)
            prob_energy = np.mean(w_coef ** 2, 1) / sum(np.mean(w_coef ** 2, 1))
            we_section = -sum(prob_energy * np.log(prob_energy))
            if we_section < base_thrd * wavelet_entropy:
                baseline_window[in_idx[i] - 1:en_idx[i]] = 1
        del prob_energy, w_coef
            
        return baseline_window
    

    def _compute_thrd(self, data_len, rms, baseline_window, base_min, epoch_time, sample_freq, thrd_perc, min_win, epo_CHF, per_CHF):
        """Compute threshold
        
        Parameters
        ----------
        data_len : int
            length of data
        rms : numpy array, shape (n_samples,)
            rms of data
        baseline_window : numpy array of 0 and 1, shape (n_samples, 1)
            baseline window
        base_min : float or int
            Baseline minimum time
        epoch_time : float or int
            Cycle Time
        sample_freq : int or float
            Sampling frequency
        thrd_perc : float or int
            Threshold percentile
        min_win : float or int
            Minimum HFO Time   
        epo_CHF : float or int
            Continous High Frequency Epoch
        per_CHF : float or int
            Continous High Frequency Percentil Threshold  
        
        Returns
        -------
        thrd : numpy array
        """
        # check data
        if rms.ndim != 1:
            raise ValueError('rms must be a 1D array')
        if baseline_window.ndim != 2 and baseline_window.shape[1] != 1:
            raise ValueError('baseline_window must be a 2D array with shape (n_samples, 1)')
        if not np.equal(baseline_window, 0).any() or not np.equal(baseline_window, 1).any():
            raise ValueError('baseline_window contain only 0 and 1')
        if len(rms) != data_len and len(baseline_window) != data_len:
            raise ValueError('rms and baseline_window must have the same length as data_len')

        # compute threshold
        wind_samples = (base_min / 60) * data_len 

        if np.sum(baseline_window) >= wind_samples:
            thrd = pos_baseline(baseline_window, rms, epoch_time, sample_freq, thrd_perc)
        else:
            thrd = neg_baseline(rms, min_win, sample_freq, epo_CHF, per_CHF)

        return thrd
    

    def _get_HFOs(self, rms, thrd, min_win, sample_freq, min_gap):
        """Get HFOs
        
        Parameters
        ----------
        rms : numpy array of float, shape (n_samples,)
            rms
        thrd : numpy array of float, shape (n_samples,)
            threshold
        min_win : int|float
            Minimum HFO Time     
        sample_freq : int|float
            sampling frequency
        min_gap : int|float
            Minimum HFO Gap
        
        Returns
        -------
        HFOs : numpy array of int, shape (n_HFOs, 2)
        """
        # check inputs
        if rms.ndim != 1:
            raise ValueError('rms must be a 1D array')
        if thrd.ndim != 1:
            raise ValueError('thrd must be a 1D array')
        if len(rms) != len(thrd):
            raise ValueError('rms and thrd must have the same length')
 
        # init outputs
        HFOs = []
        
        # get HFOs
        min_win = min_win * sample_freq
        energy_thrd = np.zeros(len(rms)+2)
        energy_thrd[1:-1] = rms >= thrd
        wind_jumps = np.diff(energy_thrd)
        wind_jump_up = np.where(wind_jumps == 1)[0]
        wind_jump_down = np.where(wind_jumps == -1)[0]
        wind_dist = wind_jump_down - wind_jump_up

        wind_select = np.where(wind_dist > min_win)[0]

        if wind_select.size == 0:
            HFO_events = []
            print('Event Selection - No detected')
            return 

        wind_ini = wind_jump_up[wind_select]
        wind_end = wind_jump_down[wind_select]
        min_gap = min_gap * sample_freq

        del wind_jumps, wind_jump_up, wind_jump_down, wind_select

        while True:
            if wind_ini.size == 0:
                HFO_events = []
                print('Event Selection - No detected')
                return

            if len(wind_ini) < 2:
                break

            next_ini = wind_ini[1:]
            last_end = wind_end[:-1]
            wind_idx = (next_ini - last_end) < min_gap

            if np.sum(wind_idx) == 0:
                break

            new_end = wind_end[1:]
            last_end[wind_idx] = new_end[wind_idx]
            wind_end[:-1] = last_end

            v_idx = np.zeros(len(wind_end)+1)
            v_idx[1:] = wind_end
            v_idx = np.diff(v_idx) != 0
            wind_ini = wind_ini[v_idx]
            wind_end = wind_end[v_idx]
        
        wind_ini = wind_ini.reshape(-1, 1)
        wind_end = wind_end.reshape(-1, 1)
        HFO_events = np.hstack((wind_ini + 1, wind_end + 1)).tolist()

            
        return HFO_events
