#!/usr/bin/env python

"""Plots smoothed JME variations for selected signal hypotheses."""

import argparse
import itertools
import math
import os
import re

import numpy as np

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


class Reader:
    """Auxiliary class to read templates from ROOT file."""
    
    def __init__(self, path):
        
        self.templates_file = ROOT.TFile(path)
    
    
    def __getitem__(self, channel_name):
        
        channel, name = channel_name
        return self._hist_to_np(self.templates_file.Get('{}/{}'.format(channel, name)))
    
    
    def _hist_to_np(self, hist):
        
        counts = np.empty(hist.GetNbinsX())
        
        for i in range(len(counts)):
            counts[i] = hist.GetBinContent(i + 1)
        
        return counts
    


if __name__ == '__main__':
    
    mpl.rc('xtick', top=True, direction='in')
    mpl.rc('ytick', right=True, direction='in')
    mpl.rc('axes', labelsize='large')
    mpl.rc('axes.formatter', limits=[-3, 4], use_mathtext=True)
    mpl.rc('errorbar', capsize=2)
    mpl.rc('lines', markersize=4)
    
    ROOT.gROOT.SetBatch(True)
    
    
    arg_parser = argparse.ArgumentParser(__doc__)
    arg_parser.add_argument('original', help='ROOT file with original templates')
    arg_parser.add_argument('smoothed', help='ROOT file with smoothed templates')
    arg_parser.add_argument(
        '-s', '--sgn', default='ggA_pos-sgn-25pc-M750', help='Name of nominal signal template'
    )
    arg_parser.add_argument(
        '-c', '--channel', default='ljets', help='Channels to process'
    )
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
    
    
    # Construct pretty label for signal process
    sgn_name_regex = re.compile('gg([AH])_((pos|neg)-(sgn|int))-(.+)pc-M(\\d+)')
    match = sgn_name_regex.match(args.sgn)
    
    if not match:
        raise RuntimeError('Failed to parse signal template name "{}".'.format(args.sgn))
    
    sgn_part_labels = {'pos-sgn': 'R', 'pos-int': 'I^{+}', 'neg-int': 'I^{-}'}
    sgn_label = 'CP-{}, $m = {:g}$ GeV, $\\Gamma / m = {:g}$%, ${}$'.format(
        'odd' if match.group(1) == 'A' else 'even',
        float(match.group(6)), float(match.group(5).replace('p', '.')),
        sgn_part_labels[match.group(2)]
    )
    
    
    # Selected channels and pretty labels for them
    if args.channel == 'ljets':
        channels = ['mujets', 'ejets']
    elif args.channel == 'll':
        channels = ['ll']
    else:
        raise RuntimeError('Channel "{}" is not supported.'.format(args.channel))
    
    channel_labels = {
        'mujets': r'$\mu + \mathrm{jets}$', 'ejets': r'$e + \mathrm{jets}$', 'll': r'\ell\ell'
    }
    
    
    # Label for x axis depends on the channel
    if args.channel == 'ljets':
        xaxis_label = r'$m_{t\bar t} \otimes \cos\/\theta^*_{\mathrm{t}_\ell}$ bin index'
    else:
        xaxis_label = r'$m_{t\bar t} \otimes c_{\mathrm{hel}}$ bin index'
    
    
    # Read bandwidths
    bandwidths = {}
    
    with open(args.bandwidths) as f:
        line = f.readline()
        
        # Skip the header of the CSV file
        while line.startswith('#'):
            line = f.readline()
        
        while line:
            tokens = line.split(',')
            bandwidths[tokens[0], tokens[1]] = (float(tokens[2]), float(tokens[3]))
            line = f.readline()
    
    
    # Variations that have been smoothed.  Only these will be plotted.
    syst_names = []
    
    for jec_syst in [
        'AbsoluteStat', 'AbsoluteScale', 'AbsoluteMPFBias', 'Fragmentation',
        'SinglePionECAL', 'SinglePionHCAL', 'FlavorQCD', 'TimePtEta',
        'RelativeJEREC1', 'RelativePtBB', 'RelativePtEC1', 'RelativeBal', 'RelativeFSR',
        'RelativeStatFSR', 'RelativeStatEC',
        'PileUpDataMC', 'PileUpPtRef', 'PileUpPtBB', 'PileUpPtEC1'
    ]:
        syst_names.append(('JEC' + jec_syst, 'CMS_scale_j_13TeV_' + jec_syst))
    
    syst_names.append(('JER', 'CMS_res_j_13TeV'))
    syst_names.append(('METUncl', 'CMS_METunclustered_13TeV'))
    
    
    
    reader_original = Reader(args.original)
    reader_smoothed = Reader(args.smoothed)
    num_bins_angle = 5
    
    for channel in channels:
        nominal = reader_original[channel, args.sgn]
        empty_bins = np.arange(len(nominal))[nominal == 0]
        
        for syst_write_name, syst_read_name in syst_names:
            for direction in ['up', 'down']:
                syst_original = reader_original[
                    channel, '{}_{}{}'.format(args.sgn, syst_read_name, direction.capitalize())
                ]
                syst_smoothed = reader_smoothed[
                    channel, '{}_{}{}'.format(args.sgn, syst_read_name, direction.capitalize())
                ]
                
                with np.errstate(all='ignore'):
                    deviation_original = syst_original / nominal - 1
                    deviation_smoothed = syst_smoothed / nominal - 1
                
                
                fig = plt.figure()
                fig.patch.set_alpha(0.)
                axes = fig.add_subplot(111)
                
                x = np.array(range(len(deviation_original) + 1))
                axes.step(
                    x, list(deviation_original) + [0], where='post',
                    solid_joinstyle='miter', color='black', label='Original'
                )
                axes.step(
                    x, list(deviation_smoothed) + [0], where='post',
                    solid_joinstyle='miter', color='C1', label='Smoothed'
                )
                
                # Mark empty bins in nominal distribution
                for i in empty_bins:
                    axes.axvspan(i, i + 1, color='0.75', zorder=2.5)
                
                axes.axhline(0., color='gray', lw=0.8, ls='dashed', zorder=2.6)
                
                for i in range(num_bins_angle - 1):
                    axes.axvline(
                        len(deviation_original) / num_bins_angle * (i + 1),
                        color='gray', lw=0.8, ls='dashed', zorder=2.6
                    )
                
                axes.margins(x=0.)
                axes.legend(loc='upper right')
                
                ylim = axes.get_ylim()
                axes.set_ylim(max(ylim[0], -0.2), min(ylim[1], 0.2))
                
                axes.set_xlabel(xaxis_label)
                axes.set_ylabel('Relative deviation from nominal')
                
                axes.text(0.01, 0.98, sgn_label, ha='left', va='top', transform=axes.transAxes)
                axes.text(
                    0., 1.003,
                    '{} ({}), {}'.format(
                        syst_write_name, direction, channel_labels[channel]
                    ),
                    ha='left', va='bottom', transform=axes.transAxes
                )
                axes.text(
                    1., 1.003,
                    '$h = ({:g}, {:g})$'.format(*bandwidths[args.sgn, syst_read_name]),
                    ha='right', va='bottom', transform=axes.transAxes
                )
                
                fig.savefig(os.path.join(
                    args.fig_dir, '{}_{}_{}_{}.pdf'.format(
                        args.sgn, syst_write_name, channel, direction
                    )
                ))
                plt.close(fig)
