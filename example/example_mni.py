import numpy as np
import pandas as pd
from HFODetector import mni
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNI wrapper.')
    parser.add_argument('-edf_path','--edf_path', type=str, required=True)
    parser.add_argument('-sample_freq','--sample_freq', type=int, required=True, help='sample frequency')
    parser.add_argument('-fp','--pass_band', type=int, required=False, help='passband', default=80)
    parser.add_argument('-fs','--stop_band', type=int, required=False, help='stop_band', default=500)
    parser.add_argument('-et','--epoch_time', type=float, required=False, help='epoch_time', default=10)
    parser.add_argument('-ec','--epo_CHF', type=float, required=False, help='epo_CHF', default=60)
    parser.add_argument('-pc','--per_CHF', type=float, required=False, help='per_CHF', default=95/100)
    parser.add_argument('-mw','--min_win', type=float, required=False, help='min_win', default=10*1e-3)
    parser.add_argument('-mg','--min_gap', type=float, required=False, help='min_gap', default=10*1e-3)
    parser.add_argument('-tp','--thrd_perc', type=float, required=False, help='threshod_percentage', default=99.9999/100)
    parser.add_argument('-bs','--base_seg', type=float, required=False, help='base_segment', default=125*1e-3)
    parser.add_argument('-bh','--base_shift', type=float, required=False, help='base_shift', default=0.5)
    parser.add_argument('-bt','--base_thrd', type=float, required=False, help='base_threshold', default=0.67)
    parser.add_argument('-bm','--base_min', type=float, required=False, help='base_min', default=5)
    parser.add_argument('-n_jobs','--n_jobs', type=int, required=False, help='n_jobs in multi-processing', default=8)
    parser.add_argument('-fn','--front_num', type=int, required=False, help='front_num', default=1)
    parser.add_argument('-save_path','--save_path', type=str, required=False, help='save_path', default=None)
    args = parser.parse_args()

    edf_path = args.edf_path
    detector = mni.MNIDetector(sample_freq=args.sample_freq, filter_freq=[args.pass_band, args.stop_band],epoch_time=args.epoch_time,
                epo_CHF=args.epo_CHF,per_CHF=args.per_CHF,min_win=args.min_win,min_gap=args.min_gap,
                thrd_perc=args.thrd_perc,base_seg=args.base_seg,base_shift=args.base_shift,
                base_thrd=args.base_thrd,base_min=args.base_min,n_jobs=args.n_jobs,front_num=args.front_num)

    channel_names, start_end = detector.detect_edf(edf_path)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    start_end = np.concatenate(start_end)
    HFO_mni_df = pd.DataFrame({"channel":channel_names,"start":start_end[:,0],"end":start_end[:,1]})
    if args.save_path is not None:
        HFO_mni_df.to_csv(args.save_path,index=False)
    else:
        HFO_mni_df.to_csv("HFO_mni.csv",index=False)