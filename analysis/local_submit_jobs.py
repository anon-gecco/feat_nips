from glob import glob
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit long jobs.",
                                     add_help=False)
    parser.add_argument('DATA_PATH',type=str)
    parser.add_argument('--r',action='store_true',dest='R', default=False)
    parser.add_argument('--c',action='store_true',dest='C', default=False)
    parser.add_argument('-n_trials',action='store',dest='TRIALS', default=1)
    parser.add_argument('-ml',action='store',dest='mls', type=str, 
            default='FeatLex,FeatLexNSGA2,FeatNSGA2,FeatRandom,FeatSimAnn,MLP')

    args = parser.parse_args()

datapath = args.DATA_PATH 

lpc_options = '-n_jobs 20'

if args.R:
    mls = ','.join([ml + 'R' for ml in args.mls.split(',')])
    for f in glob(datapath + "/regression/*/*.tsv.gz"):
        jobline =  ('python analyze.py {DATA} '
                   '-ml {ML} '
                   '-results ../results_local/regression -n_trials {NT} {LPC}').format(DATA=f,
                                                          LPC=lpc_options,
                                                          ML=mls,
                                                          NT=args.TRIALS)
 
        print(jobline)
        os.system(jobline)

if args.C:
    mls = ','.join([ml + 'C' for ml in args.mls.split(',')])
    for i,f in enumerate(glob(datapath + "/classification/*/*.tsv.gz")):
#    if i==0:
        jobline =  ('python analyze.py {DATA} '
               '-ml {ML} '
               '-results ../results_local/classification -n_trials {NT} {LPC}').format(DATA=f,
                                                      LPC=lpc_options,
                                                      ML=mls,
                                                      NT=args.TRIALS)
        print(jobline)
        os.system(jobline)

