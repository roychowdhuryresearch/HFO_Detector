
# HFODetector
[![PyPI version](https://badge.fury.io/py/hfodetector.svg)](https://badge.fury.io/py/hfodetector)   ![PyPI - Downloads](https://img.shields.io/pypi/dm/HFODetector)  ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/roychowdhuryresearch/HFO_Detector)



HFODetector is Python package that that is capable of detecting HFOs with either a STE or an MNI detector. Detection speed is increased by using multiprocessing.

### Bibtex 
If you find our project is useful in your research, please cite:
```
Zhang, Y., Liu, L., Ding, Y., Chen, X., Monsoor, T., Daida, A., ... & Roychowdhury, V. (2023). PyHFO: Lightweight Deep Learning-powered End-to-End High-Frequency Oscillations Analysis Application. bioRxiv, 2023-08.
```


## Installation
```
pip install HFODetector
```

------- 
## Run time comparison (in minutes)

|  | Linux  | Linux | Windows | Windows | OS X | OS X |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  | STE | MNI | STE | MNI | STE | MNI |
| RIPPLELAB | 372.83 | 5647.12 | - | - | - | - |
| pyHFO single-core | 57.43 | 971.35 | 34.57 | 933.31 | 35.90 | 659.63 |
| pyHFO multi-core  | 5.18 | 83.30 | 9.03 | 113.59 | 7.56 | 114.35 |

The testing data we are using is 19 patients 10 min data in [Refining epileptogenic high-frequency oscillations using deep learning: A novel reverse engineering approach](https://academic.oup.com/braincomms/article/4/1/fcab267/6420212) paper. 

** Single-core: n-jobs =1 for all machines,

** Multi-core: n-jobs = 32 for Linux (AMD Ryzen Threadripper 2950X), n-jobs = 8 for Windows(Intel i9-13900K) and Mac machines(Apple M1 Pro).

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
    channel_names, start_end = detector.detect_edf(edf_path)
    # channel_names is a list that is the same length as the number of channels in the edf
    # start_end is a nested list with the same length as channel_names. start_end[i][j][0] and start_end[i][j][1] 
    # will give the start and end index respectively for the jth detected HFO in channel channel_names[i]
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    start_end = np.concatenate(start_end)
    HFO_ste_df = pd.DataFrame({"channel":channel_names,"start":start_end[:,0],"end":start_end[:,1]})
```
Which results a pandas dataframe `HFO_ste_df` a sample is displayed below:

<img src="https://github.com/roychowdhuryresearch/HFO_Detector/blob/main/img/readme/HFO_ste_df_sample.png" width="200">

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
    # channel_names is a list that is the same length as the number of channels in the edf
    # start_end is a nested list with the same length as channel_names. start_end[i][j][0] and start_end[i][j][1] 
    # will give the start and end index respectively for the jth detected HFO in channel channel_names[i]
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    start_end = np.concatenate(start_end)
    HFO_mni_df = pd.DataFrame({"channel":channel_names,"start":start_end[:,0],"end":start_end[:,1]})
```
Which results a pandas dataframe `HFO_mni_df` a sample is displayed below:

<img src="https://github.com/roychowdhuryresearch/HFO_Detector/blob/main/img/readme/HFO_mni_df_sample.png" width="200">


This dataframe has the following 3 columns:
- `channel` : name of the channel corresponding to the detected HFO
- `start` : start timestamp of the detected HFO in milliseconds
- `end` : end timestamp of the detected HFO in milliseconds

### Contributors:
Department of Electrical and Computer Engineering, University of California, Los Angeles
- [Xin Chen](https://www.linkedin.com/in/xin-chen-980521/) -- Main Author of MNI
- [Hoyoung Chung](https://www.linkedin.com/in/tc01/) -- Main Author of STE
- [Lawrence Liu](https://www.linkedin.com/in/lawrence-liu-0a01391a7/)
- [Qiujing Lu](https://www.linkedin.com/in/qiujing-lu-309042126/)
- [Yuanyi Ding](https://www.linkedin.com/in/yuanyi-ding-4a981a132/)
- [Yipeng Zhang](https://zyp5511.github.io/)

Division of Pediatric Neurology, Department of Pediatrics, UCLA Mattel Children’s Hospital David Geffen School of Medicine
- [Hiroki Nariai](https://www.uclahealth.org/providers/hiroki-nariai)


### Contacts:
Please create a github issue or email the following people for further information 
- Lawrence Liu (lawrencerliu@g.ucla.edu) 
- Yuanyi Ding (semiswiet@g.ucla.edu)
