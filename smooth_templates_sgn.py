#!/usr/bin/env python

"""Applies smoothing to JME variations in signal templates."""

import argparse
import itertools
import math
import os
import re

import numpy as np

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from smoothutils import AdaptiveRebinner, Reader, Smoother


def create_TH1(template, label, name):
    nbins = len(template)
    hist = ROOT.TH1D(label, name, nbins, 0, nbins)
    for i in range(nbins):
        hist.SetBinContent(i + 1, template[i, 0])
        hist.SetBinError(i + 1, math.sqrt(template[i, 1]))
    return hist


if __name__ == '__main__':
    
    ROOT.gROOT.SetBatch(True)
    
    arg_parser = argparse.ArgumentParser(__doc__)
    arg_parser.add_argument('input', help='ROOT file with original templates')
    arg_parser.add_argument(
        '-b', '--bandwidths', default='bandwidths.csv',
        help='CSV file with chosen bandwidths'
    )
    arg_parser.add_argument(
        '-o', '--output', default='templates.root',
        help='Name for output ROOT file with smoothed variations in signal'
    )
    args = arg_parser.parse_args()
    
    
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
    
    
    src_file = ROOT.TFile(args.input)
    out_file = ROOT.TFile(args.output, 'recreate')

    reader = Reader(args.input)
    
    for sys_name, bandwidth in bandwidths.items():
        template_nominal = reader.read_counts(sys_name[0])
        rebinner = AdaptiveRebinner(template_nominal)
        
        template_up = reader.read_counts(sys_name[0] + "_" + sys_name[1] + "_up")
        template_down = reader.read_counts(sys_name[0] + "_" + sys_name[1] + "_down")
        
        smoother = Smoother(
            template_nominal, template_up, template_down,
            rebinner, rebin_for_smoothing=True
        )
        template_up_smooth, template_down_smooth = smoother.smooth(bandwidth * reader.num_bins)
        create_TH1(template_up_smooth, sys_name[0] + "_" + sys_name[1] + "_up", sys_name[0] + "_" + sys_name[1] + "_up").Write()
        create_TH1(template_down_smooth, sys_name[0] + "_" + sys_name[1] + "_down", sys_name[0] + "_" + sys_name[1] + "_down").Write()
        
    
    out_file.Close()
    src_file.Close()
