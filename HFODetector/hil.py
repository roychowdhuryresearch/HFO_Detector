import numpy as np
from scipy.signal import hilbert
from .utils import *
from scipy.io import savemat #改动

class HILDetector():
    
    def __init__(self, sample_freq, filter_freq=[80, 500], 
                sd_thres=5, min_window=6 * 1e-3, 
                epoch_len=600, n_jobs=32, 
                use_kwargs=True, front_num=1):
        """
        Initialize the HIL-based HFO detector.
        
        Parameters
        ----------
        sample_freq : float | int
            Sampling frequency of the data in Hz
        filter_freq : list of float or int, length 2, default [80, 500] Hz
            Filter freqs in Hz
        sd_thres : float | int, default 5
            Threshold in standard deviations for HFO detection
        min_window : float | int, default 6 * 1e-3 s
            Minimum window time for an HFO in seconds(s)
        epoch_len : float | int, default 600 s
            Cycle time in seconds(s)
        n_jobs : int, default 32
            Number of jobs to run in parallel
        use_kwargs : bool, default True
            Whether to pass additional arguments to the map function
        front_num : int, default 1
            Number of jobs to run in parallel at the front end
        """
        print("starting detection...")
        sample_freq = validate_param(sample_freq, 'sample_freq', (int, float))
        filter_freq = validate_filter_freq(filter_freq)
        if filter_freq[1] > sample_freq / 2:
            raise ValueError('filter_freq[1] must be less than sample_freq/2')
        sd_thres = validate_param(sd_thres, 'sd_thres', (int, float))
        min_window = validate_param(min_window, 'min_window', (int, float))
        epoch_len = validate_param(epoch_len, 'epoch_len', (int, float))
        n_jobs = validate_param(n_jobs, 'n_jobs', int)
        use_kwargs = validate_param(use_kwargs, 'use_kwargs', bool)
        front_num = validate_param(front_num, 'front_num', int)
        
        self.sample_freq = sample_freq
        self.filter_freq = filter_freq
        self.sd_thres = sd_thres
        self.min_window = min_window
        self.epoch_len = epoch_len
        self.n_jobs = n_jobs
        self.use_kwargs = use_kwargs
        self.front_num = front_num

    def detect_edf(self, edf_path):
        """
        Detect HFOs from an EDF file.

        Parameters
        ----------
        edf_path : str
            Path to the EDF file

        Returns
        -------
        channel_names : numpy array of str, shape (n_channels,)
        HFOs : numpy array of int, shape (n_channels, n_HFOs, 2)
        """
        edf_path = validate_type(edf_path, 'edf_path', str)
        raw, channels = read_raw(edf_path)
        return self.detect_multi_channels(data=raw, channels=channels)
    
    def detect_multi_channels(self, data, channels, filtered=False):
        """
        Detect HFOs from numpy arrays of multi channels in parallel.

        Parameters
        ----------
        data : numpy array like of float, shape (n_channels, n_samples)
        channels : numpy array like of str, shape (n_channels,)
        filtered : bool, default False
            Whether the data is already filtered

        Returns
        -------
        channel_names : numpy array of str, shape (n_channels,)
        HFOs : numpy array of int, shape (n_channels, n_HFOs, 2)
        """
        data = validate_type(data, 'data', np.ndarray)
        channels = validate_type(channels, 'channels', np.ndarray)

        # 只处理第一个通道
        # first_channel_data = data[0]
        # first_channel_name = channels[0]

        n_channels = data.shape[0]  # 获取通道数
        channel_HFOs = []
        
        for i in range(n_channels):
            print(f"Processing channel {channels[i]}...")
            HFOs, channel_name = self.detect(data[i], channels[i], filtered)
            if HFOs is not None:
                channel_HFOs.append(HFOs)
            else:
                channel_HFOs.append([])  # 如果没有找到HFO，则追加一个空列表
    
        return np.array(channels), np.array(channel_HFOs, dtype=object)


    def detect(self, data, channel_names, filtered=False):
        """
        Detect HFOs from a single channel.

        Parameters
        ----------
        data : numpy array like of float, shape (n_samples,)
        channel_names : str
        filtered : bool, default False
            Whether the data is already filtered

        Returns
        -------
        HFOs : numpy array of int, shape (n_HFOs, 2)
        channel_names : str
        """
        data = validate_type(data, 'data', np.ndarray)
        if data.shape[0] == 0:
            return None, None
        # if data.ndim != 1:
        #     raise ValueError('data must be a 1D array')
        channel_names = validate_type(channel_names, 'channel_names', str)
        
        # Preprocessing Filter
        if not filtered:
            try:
                data = preprocess(data, self.sample_freq, self.filter_freq)
            except Exception as e:
                print(f"Error during saving: {e}")
        
        
        # Hilbert Transform Calculation
        hilbert_transformed = np.abs(hilbert(data))
        savemat('hilbert_signal_python.mat', {'hilbert_signal': hilbert_transformed})
        # Thresholding 
        epoch_lims = self._compute_epoch_lims(len(data), self.sample_freq, self.epoch_len)
        epoch_lims = epoch_lims + 1  # 调整起始点，确保与 MATLAB 对齐
        epoch_lims = np.unique(epoch_lims, axis=0)  # 删除重复的行
        HFOs = self._get_HFOs(hilbert_transformed, epoch_lims, self.sample_freq, self.min_window, self.sd_thres)
        return HFOs, channel_names

    def _compute_epoch_lims(self, data_len, sample_freq, epoch_len):
        """
        Compute the limits of each epoch.

        Parameters
        ----------
        data_len : int | float
        sample_freq : float | int
        epoch_len : int | float

        Returns
        -------
        epoch_lims : numpy array of int, shape (n_epochs, 2)
        """
        epoch_len = round(epoch_len * sample_freq)
        temp = np.arange(0, data_len, epoch_len)

        if temp[-1] < data_len:
            temp = np.append(temp, data_len)
        epoch_lims = np.vstack([temp[:-1], temp[1:]]).T
        return epoch_lims
    
    # def _get_HFOs(self, hilbert_transformed, epoch_lims, sample_freq, min_window, sd_thres):
    #     """
    #     Get HFOs from a single channel.
    #     """
    #     min_window = round(min_window * sample_freq)

    #     HFOs = []
    #     for i, j in epoch_lims:
    #         epoch_data = hilbert_transformed[i:j]
    #         mean_val = np.mean(epoch_data)
    #         std_val = np.std(epoch_data)
    #         thresholded = epoch_data > (mean_val + sd_thres * std_val)
            
    #         # Selection of Valid Intervals
    #         wind_thres = np.pad(thresholded.astype(int), 1)
    #         wind_jumps = np.diff(wind_thres)
    #         wind_jump_up = np.where(wind_jumps == 1)[0]
    #         wind_jump_down = np.where(wind_jumps == -1)[0] - 1
    #         wind_dist = wind_jump_down - wind_jump_up

    #         valid_intervals = wind_dist > min_window
    #         wind_ini = wind_jump_up[valid_intervals]
    #         wind_end = wind_jump_down[valid_intervals]

    #         if len(wind_ini) == 0:
    #             continue

    #         wind_intervals = np.vstack((wind_ini, wind_end)).T
    #         HFOs.extend(wind_intervals)

    #     return np.array(HFOs)

    def _get_HFOs(self, hilbert_transformed, epoch_lims, sample_freq, min_window, sd_thres):
        """
        Get HFOs from a single channel.
        
        Parameters
        ----------
        hilbert_transformed : numpy array of float
            Hilbert transform of the signal.
        epoch_lims : numpy array of int
            Limits of each epoch, each row is [start, end].
        sample_freq : float or int
            Sampling frequency of the signal.
        min_window : float
            Minimum window time for an HFO in seconds.
        sd_thres : float
            Threshold in standard deviations for HFO detection.
        
        Returns
        -------
        numpy array of int
            Detected HFOs as [start, end] intervals.
        """
        
        # Convert min_window to sample points
        min_window = round(min_window * sample_freq)
        
        HFOs = []
        
        # Iterate through each epoch defined by epoch_lims
        for epoch_idx, (i, j) in enumerate(epoch_lims):
            
            # Extract the signal for the current epoch
            epoch_data = hilbert_transformed[i:j]
            
            # Compute mean and standard deviation for the epoch
            mean_val = np.mean(epoch_data)
            std_val = np.std(epoch_data)
            
            # Compute the threshold
            threshold = mean_val + sd_thres * std_val
            
            # Thresholding the epoch data
            thresholded = epoch_data > threshold
            
            # Convert thresholded signal to 1s and 0s and add padding
            wind_thres = np.pad(thresholded.astype(int), 1)
            
            # Detect rising and falling edges of threshold crossings
            wind_jumps = np.diff(wind_thres)
            wind_jump_up = np.where(wind_jumps == 1)[0]
            wind_jump_down = np.where(wind_jumps == -1)[0] - 1
            
            # Calculate the distances between jump up and jump down points
            wind_dist = wind_jump_down - wind_jump_up
            
            # Select valid intervals where the distance exceeds the minimum window length
            valid_intervals = wind_dist > min_window
            wind_ini = wind_jump_up[valid_intervals]
            wind_end = wind_jump_down[valid_intervals]
            
            # If no valid intervals are found, skip to the next epoch
            if len(wind_ini) == 0:
                print(f"No valid HFOs found in Epoch [{i}:{j}]")
                continue
            
            # Combine valid intervals into a list of HFOs
            wind_intervals = np.vstack((wind_ini, wind_end)).T
            
            # fixing the start and end time of each HFO
            wind_intervals = np.vstack((wind_ini + i, wind_end + i)).T
            
            # Add detected intervals to the HFO list
            HFOs.extend(wind_intervals)
        
        # Convert the list of HFOs to a numpy array and return
        HFOs_array = np.array(HFOs)
        
        return HFOs_array


