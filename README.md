# HFODetector

HFODetector is Python package that that is capable of detecting HFOs with either a STE or an MNI detector. Detection speed is increased by using multiprocessing.

## Installation
```pip install HFODetector```

## Example usage 
### STE detector
To use the STE detector, import `ste` from `HFODetector`, then a detector can be initialized with the desired parameters by calling `ste.STEDetector`. To use it to detect HFOs from an `.edf` file, call the `detect_edf` method. The `detect_edf` method takes a path to an edf file as input and returns a tuple containing the channel names and start and end timestamps of detected HFOs. The following code snippet shows how to use the STE detector.
```
import numpy as np
import pandas as pd
from HFODetector import ste

if __name__ == "__main__":
    edf_path = "example.edf" #change this to your edf path
    detector = ste.STEDetector(sample_freq=2000, filter_freq=[80, 500], 
                rms_window=3*1e-3, min_window=6*1e-3, min_gap=10 * 1e-3, 
                epoch_len=600, min_osc=6, rms_thres=5, peak_thres=3,
                n_jobs=32, front_num=1)
    ## channel_names will be the same length as the channels in the edf
    ## start_end will be a nested 2D list which is the same length as the channel_names and 
    ## contains start and end indexs of each HFOs in each channel.
    channel_names, start_end = detector.detect_edf(edf_path)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    start_end = np.concatenate(start_end)
    HFO_ste_df = pd.DataFrame({"channel":channel_names,"start":start_end[:,0],"end":start_end[:,1]})
```
Which results a pandas dataframe `HFO_ste_df` a sample is displayed below:

![image](HFO_ste_df_sample.png)

This dataframe has the following 3 columns:
- `channel` : name of the channel corresponding to the detected HFO
- `start` : start timestamp of the detected HFO in milliseconds
- `end` : end timestamp of the detected HFO in milliseconds


### MNI detector
To use the MNI detector, import `mni` from `HFODetector`, then a detector can be initialized with the desired parameters by calling `mni.MNIDetector`. To use it to detect HFOs from an `.edf` file, call the `detect_edf` method. The `detect_edf` method takes a path to an edf file as input and returns a tuple containing the channel names and start and end timestamps of detected HFOs. The following code snippet shows how to use the MNI detector.
```
import numpy as np
import pandas as pd
from HFODetector import mni

if __name__ == "__main__":
    edf_path = "example_edf.edf" #change this to your edf path
    sample_freq=2000 #change this to your sample frequency
    detector = mni.MNIDetector(sample_freq, filter_freq=[80, 500], 
                epoch_time=10, epo_CHF=60, per_CHF=95/100, 
                min_win=10*1e-3, min_gap=10*1e-3, thrd_perc=99.9999/100, 
                base_seg=125*1e-3, base_shift=0.5, base_thrd=0.67, base_min=5,
                n_jobs=32, front_num=1)
    channel_names, start_end = detector.detect_edf(edf_path)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    start_end = np.concatenate(start_end)
    HFO_mni_df = pd.DataFrame({"channel":channel_names,"start":start_end[:,0],"end":start_end[:,1]})
```
Which results a pandas dataframe `HFO_mni_df` a sample is displayed below:

![image](HFO_mni_df_sample.png)

This dataframe has the following 3 columns:
- `channel` : name of the channel corresponding to the detected HFO
- `start` : start timestamp of the detected HFO in milliseconds
- `end` : end timestamp of the detected HFO in milliseconds

### Contributors:
Deparement of Electrical and Computer Engineering, University of California, Los Angeles
- [Xin Chen](https://www.linkedin.com/in/xin-chen-980521/)
- [Hoyoung Chung](https://www.linkedin.com/in/tc01/)
- [Lawrence Liu](https://www.linkedin.com/in/lawrence-liu-0a01391a7/)
- [Qiujing Lu](https://www.linkedin.com/in/qiujing-lu-309042126/)
- [Yipeng Zhang](https://zyp5511.github.io/)

Division of Pediatric Neurology, Department of Pediatrics, UCLA Mattel Children’s Hospital David Geffen School of Medicine
- [Hiroki Nariai](https://www.uclahealth.org/providers/hiroki-nariai)


### Contracts:
Please create github issue or email the following people for further information 
- Lawrence Liu (lawrencerliu@g.ucla.edu) 
- Yuanyi Ding (semiswiet@g.ucla.edu)