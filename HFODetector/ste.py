import numpy as np
from scipy.signal import *
from .utils import *

class STEDetector():
    
    def __init__(self, sample_freq, filter_freq=[80, 500], 
                rms_window=3 * 1e-3, min_window=6 * 1e-3, min_gap=10 * 1e-3, 
                epoch_len=600, min_osc=6, rms_thres=5, peak_thres=3,
                n_jobs=32, use_kwargs=True, front_num=1):
        """Initialize the RMS detector
        sample_freq : float | int
            Sampling frequency of the data in Hz
        filter_freq : list of float or int, length 2, default [80, 500] Hz
            Filter freqs in Hz
        rms_window : float | int, default 3 * 1e-3 s
            RMS window time in milliseconds(ms)
        min_window : float | int, default 6 * 1e-3 s
            Minimum window time for an HFO in milliseconds(ms)
        min_gap : float | int, default 10 * 1e-3 s
            Minimum distance time between two HFO candidates in milliseconds(ms)
        epoch_len : float | int, default 600 s
            Cycle time in seconds(s)
        min_osc : float | int, default 6
            Minimum oscillations per interval
        rms_thres : float | int, default 5
            RMS threshold in standard deviation
        peak_thres : float | int, default 3
            threshold for peak detection
        n_jobs : int, default 32
            Number of jobs to run in parallel
        use_kwargs : bool, default True
            Whether to pass additional arguments to the map function
        front_num : int, default 1
            Number of jobs to run in parallel at the front end
        """
        sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
        filter_freq = validate_filter_freq(filter_freq)
        if filter_freq[1] > sample_freq/2:
            raise ValueError('filter_freq[1] must be less than sample_freq/2')
        rms_window = validate_param(rms_window, 'rms_window', (int, float))
        min_window = validate_param(min_window, 'min_window', (int, float))
        min_gap = validate_param(min_gap, 'min_gap', (int, float))
        epoch_len = validate_param(epoch_len, 'epoch_len', (int, float))  
        min_osc = validate_param(min_osc, 'min_osc', (int, float))
        rms_thres = validate_param(rms_thres, 'rms_thres', (int, float))
        peak_thres = validate_param(peak_thres, 'peak_thres', (int, float))
        n_jobs = validate_param(n_jobs, 'n_jobs', int)
        use_kwargs = validate_param(use_kwargs, 'use_kwargs', bool)
        front_num = validate_param(front_num, 'front_num', int)
        
        self.sample_freq = sample_freq
        self.filter_freq = filter_freq
        self.rms_window = rms_window
        self.min_window = min_window
        self.min_gap = min_gap
        self.epoch_len = epoch_len
        self.min_osc = min_osc
        self.rms_thres = rms_thres
        self.peak_thres = peak_thres
        self.n_jobs = n_jobs
        self.use_kwargs = use_kwargs
        self.front_num = front_num


    def detect_edf(self, edf_path):
        """detect HFOs from an EDF file
        
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
        """detect HFOs from numpy arrays of multi channels parrallelly
        
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
        filtered : bool, default False
            Whether the data is already filtered

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

        if filtered == False:        
            data = preprocess(data, self.sample_freq, self.filter_freq)
        rms = compute_rms(data, self.sample_freq, self.rms_window, detector='RMS')
        epoch_lims = self._compute_epoch_lims(len(data), self.sample_freq, self.epoch_len)
        HFOs = self._get_HFOs(data, rms, epoch_lims, self.sample_freq, self.min_window, self.min_gap, self.min_osc, self.rms_thres, self.peak_thres)
        return HFOs, channel_names

    
    def _compute_epoch_lims(self, data_len, sample_freq, epoch_len):
        """
        Compute the limits of each epoch
        
        Parameters
        ----------
        data_len : int | float
        sample_freq : float | int
        epoch_len : int | float

        Returns
        -------
        epoch_lims : numpy array of int, shape (n_epochs, 2)
        """

        # compute epoch limits
        epoch_len = round(epoch_len * sample_freq)
        temp = np.arange(0, data_len, epoch_len)

        if temp[-1] < data_len:
            temp = np.append(temp, [data_len])
        epoch_lims = np.vstack([[temp[:-1].T, temp[1:]]]).T
        return epoch_lims


    def _get_HFOs(self, filtered, rms, epoch_lims, sample_freq, min_window, min_gap, min_osc, rms_thres, peak_thres):
        """
        Get HFOs from a single channel
        
        Parameters
        ----------
        filtered : numpy array of float, shape (n_samples,)
        rms : numpy array of float, shape (n_samples,)
        epoch_lims : numpy array of int, shape (n_epochs, 2)
        sample_freq : float | int
        min_window : int | float
        min_gap : int | float
        min_osc : int
        rms_thres : int | float
        peak_thres : int | float

        Returns
        -------
        HFOs : numpy array of int, shape (n_HFOs, 2)
        """
        
        # check input parameters
        if filtered.ndim != 1:
            raise ValueError('filtered must be a 1D array')
        if rms.ndim != 1:
            raise ValueError('rms must be a 1D array')
        if epoch_lims.ndim != 2:    
            raise ValueError('epoch_lims must be a 2D array')   
        if epoch_lims.shape[1] != 2:
            raise ValueError('epoch_lims must have 2 columns')
        if rms.shape[0] != filtered.shape[0]:
            raise ValueError('rms and filtered must have the same length')
        
        min_window = round(min_window * sample_freq)
        min_gap = round(min_gap * sample_freq)

        # get HFOs
        HFOs = []
        for i, j in epoch_lims:
            # calculate threshold
            window = np.zeros(len(rms))
            window[i:j] = 1
            rms_epoch = rms * window
            rms_interval = rms[i:j]
            epoch_filt = filtered[i:j]
            thres_rms = (rms_epoch > (np.mean(rms_interval) + rms_thres * np.std(rms_interval))).astype('int')      

            if len(np.argwhere(thres_rms)) == 0:
                # print("none satisfies THRES_RMS requirement")
                pass

            wind_thres = np.pad(thres_rms, 1)
            wind_jumps = np.diff(wind_thres)
            wind_jump_up = np.argwhere(wind_jumps == 1)
            wind_jump_down = np.argwhere(wind_jumps == -1)
            wind_dist = wind_jump_down - wind_jump_up
            wind_ini = wind_jump_up[wind_dist > min_window] + 1  
            wind_end = wind_jump_down[wind_dist > min_window]

            if len(wind_ini) == 0:
                # print("none satisfies MIN_WINDOW requirement")
                pass

            while True:
                next_ini = wind_ini[1:]
                last_end = wind_end[:-1]
                wind_idx = (next_ini - last_end) < min_gap

                if np.sum(wind_idx) == 0:
                    # print("break while")
                    break

                new_end = wind_end[1:]
                last_end[wind_idx] = new_end[wind_idx]
                wind_end[:-1] = last_end

                idx = np.diff(np.pad(wind_end, (1, 0))) != 0
                wind_ini = wind_ini[idx]
                wind_end = wind_end[idx]    

            wind_intervals = np.array([wind_ini, wind_end]).T

            # select intervals 
            count = 1
            wind_select = []
            thres_peak = np.mean(np.abs(epoch_filt) + peak_thres * np.std(np.abs(epoch_filt)))

            for ii, jj in wind_intervals:
                temp = np.abs(filtered[ii-1:jj])
                if len(temp) < 3:
                    continue

                peak_ind, _ = find_peaks(temp, height=thres_peak)
                if len(peak_ind) < min_osc:
                    continue

                wind_select.append([ii, jj])
                count += 1

            if len(wind_select):
                HFOs += wind_select
        return HFOs                             

