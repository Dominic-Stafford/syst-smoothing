#!/usr/bin/env python

"""Plots smoothed JME variations for selected signal hypotheses."""

import argparse
import os

import numpy as np
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import mplhep as hep
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import uproot

from smoothutils import Reader


if __name__ == '__main__':
    
    ROOT.gROOT.SetBatch(True)
    
    
    arg_parser = argparse.ArgumentParser(__doc__)
    arg_parser.add_argument('original', help='ROOT file with original templates')
    arg_parser.add_argument('smoothed', help='ROOT file with smoothed templates')
    arg_parser.add_argument(
        '-b', '--bandwidths', default='bandwidths.csv',
        help='CSV file with chosen bandwidths'
    )
    arg_parser.add_argument(
        '--fig-dir', default='fig/smoothing', help='Directory for plots'
    )
    args = arg_parser.parse_args()
    
    try:
        os.makedirs(args.fig_dir)
    except FileExistsError:
        pass

    # Read bandwidths
    bandwidths = {}
    
    with open(args.bandwidths) as f:
        line = f.readline()
        
        # Skip the header of the CSV file
        while line.startswith('#'):
            line = f.readline()
        
        while line:
            tokens = line.split(',')
            bandwidths[tokens[0], tokens[1]] = float(tokens[2])
            line = f.readline()

    reader_orig = Reader(args.original)
  
    for sys_name, bandwidth in bandwidths.items():
        nominal = reader_orig.read_counts(sys_name[0])[:, 0]
        
        for direction in ["up", "down"]:
            syst_orig = reader_orig.read_counts(sys_name[0] + "_" + sys_name[1] + "_" + direction)[:, 0]
            with uproot.open(args.smoothed) as f:
                syst_smooth, _ = f[sys_name[0] + "_" + sys_name[1] + "_" + direction].to_numpy()
            deviation_orig = syst_orig / nominal - 1
            deviation_smooth = syst_smooth / nominal - 1
            x = np.array(range(len(deviation_orig) + 1))
            fig, ax = plt.subplots()
            plt.step(x, list(deviation_orig) + [0], where="post",
                    solid_joinstyle="miter", color="tab:blue", label="Original")
            plt.step(x, list(deviation_smooth) + [0], where="post",
                    solid_joinstyle="miter", color="tab:red", label="Smoothed")
            plt.legend()
            ax.axhline(0., color='gray', lw=0.8, ls='dashed', zorder=2.6)
            ax.set_ylabel('Relative deviation from nominal')
            plt.savefig(os.path.join(args.fig_dir, sys_name[0] + "_" + sys_name[1] + "_" + direction))
            plt.clf()
