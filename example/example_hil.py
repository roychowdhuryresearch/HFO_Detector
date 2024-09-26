import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HFODetector import hil
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HIL wrapper.')
    parser.add_argument('-edf_path','--edf_path', type=str, required=True)
    parser.add_argument('-sample_freq','--sample_freq', type=int, required=True, help='Sample frequency')
    parser.add_argument('-fp','--pass_band', type=int, required=False, help='Passband', default=80)
    parser.add_argument('-fs','--stop_band', type=int, required=False, help='Stop band', default=500)
    parser.add_argument('-et','--epoch_time', type=float, required=False, help='Epoch time', default=10)
    parser.add_argument('-sw','--sd_thres', type=float, required=False, help='Standard deviation threshold', default=5)
    parser.add_argument('-mw','--min_win', type=float, required=False, help='Minimum window time for HFOs', default=6*1e-3)
    parser.add_argument('-n_jobs','--n_jobs', type=int, required=False, help='Number of jobs for multiprocessing', default=8)
    parser.add_argument('-fn','--front_num', type=int, required=False, help='Number of jobs to run in parallel at the front end', default=1)
    parser.add_argument('-save_path','--save_path', type=str, required=False, help='Save path', default=None)
    args = parser.parse_args()

    edf_path = args.edf_path
    detector = hil.HILDetector(sample_freq=args.sample_freq, filter_freq=[args.pass_band, args.stop_band],
                               sd_thres=args.sd_thres, min_window=args.min_win, 
                               epoch_len=args.epoch_time, n_jobs=args.n_jobs, front_num=args.front_num)

    channel_names, start_end = detector.detect_edf(edf_path)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i]) > 0]
    start_end = np.concatenate(start_end)
    HFO_hil_df = pd.DataFrame({"channel": channel_names, "start": start_end[:,0], "end": start_end[:,1]})
    
    if args.save_path is not None:
        HFO_hil_df.to_csv(args.save_path, index=False)
    else:
        HFO_hil_df.to_csv("HFO_hil.csv", index=False)
