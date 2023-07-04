import numpy as np
import pandas as pd
## add parent path of this file to sys.path, so we can import HFODetector
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HFODetector import ste
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='STE wrapper.')
    parser.add_argument('-edf_path','--edf_path', type=str, required=True)
    parser.add_argument('-sample_freq','--sample_freq', type=int, required=True, help='sample frequency')
    parser.add_argument('-fp','--pass_band', type=int, required=False, help='passband', default=80)
    parser.add_argument('-fs','--stop_band', type=int, required=False, help='stop_band', default=500)
    parser.add_argument('-rw','--rms_window', type=float, required=False, help='rms_window', default=3*1e-3)
    parser.add_argument('-mw','--min_window', type=float, required=False, help='min_window', default=6*1e-3)
    parser.add_argument('-mg','--min_gap', type=float, required=False, help='min_gap', default=10*1e-3)
    parser.add_argument('-el','--epoch_len', type=float, required=False, help='epoch_len', default=600)
    parser.add_argument('-mo','--min_osc', type=int, required=False, help='min_osc', default=6)
    parser.add_argument('-rt','--rms_thres', type=float, required=False, help='rms_thres', default=5)
    parser.add_argument('-pt','--peak_thres', type=float, required=False, help='peak_thres', default=3)
    parser.add_argument('-n_jobs','--n_jobs', type=int, required=False, help='n_jobs in multi-processing', default=8)
    parser.add_argument('-fn','--front_num', type=int, required=False, help='front_num', default=1)
    parser.add_argument('-save_path','--save_path', type=str, required=False, help='save_path', default=None)
    args = parser.parse_args()

    edf_path = args.edf_path
    detector = ste.STEDetector(sample_freq=args.sample_freq, filter_freq=[args.pass_band, args.stop_band], 
                rms_window=args.rms_window, min_window=args.min_window, min_gap=args.min_gap, 
                epoch_len=args.epoch_len, min_osc=args.min_osc, rms_thres=args.rms_thres, peak_thres=args.peak_thres,
                n_jobs=args.n_jobs, front_num=args.front_num)

    channel_names, start_end = detector.detect_edf(edf_path)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    start_end = np.concatenate(start_end)
    HFO_ste_df = pd.DataFrame({"channel":channel_names,"start":start_end[:,0],"end":start_end[:,1]})
    if args.save_path is not None:
        HFO_ste_df.to_csv(args.save_path, index=False)
    else:
        HFO_ste_df.to_csv("HFO_ste.csv", index=False)
