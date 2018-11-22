#!/usr/bin/env python

"""Analyses output files produced by cross_validation_sgn.py.

For each template and each variation find the bandwidth that gives the
smallest CV error.  Plots mean CV errors for requested cases.
"""

import argparse
import json
import math
import os
import sys
import re

import numpy as np

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt


def parse_file(path):
    """Parse file with output of a single job."""
    
    with open(path) as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        raise RuntimeError('File "{}" is empty.'.format(path))
    
    
    # Drop CSV header
    del lines[0]
    
    
    split_lines = [line.split(',') for line in lines]
    
    for i in range(len(split_lines)):
        if len(split_lines[i]) != 7:
            raise RuntimeError('In file "{}", failed to parse line "{}".'.format(path, lines[i]))
    
    
    # Template and variation are the same for one file.  Extract them
    # from the first line.
    template = split_lines[0][0]
    variation = split_lines[0][1]
    
    errors = [
        (float(l[2]), float(l[3]), float(l[5])) for l in split_lines
    ]
    
    return template, variation, errors


if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser(epilog=__doc__)
    arg_parser.add_argument('inputs', help='Directory with outputs of individual jobs')
    arg_parser.add_argument(
        '-o', '--output', default='bandwidths.csv',
        help='Name for output file with chosen bandwidths'
    )
    arg_parser.add_argument(
        '--plots',
        help='Comma-separated names of signal templates for which plots should be produced'
    )
    arg_parser.add_argument(
        '--fig-dir', default='fig/CV',
        help='Directory for produced figures'
    )
    args = arg_parser.parse_args()
    
    try:
        os.makedirs(args.fig_dir)
    except FileExistsError:
        pass
    
    
    # Find optimal bandwidths
    optimal_badwidths = []
    
    for file_name in os.listdir(args.inputs):
        path = os.path.join(args.inputs, file_name)
      
        if not os.path.isfile(path):
            continue
      
        try:
            template, variation, errors = parse_file(path)
            i = np.argmin(np.asarray(errors)[:, 2])
            bandwidth = (errors[i][0], errors[i][1])
          
            optimal_badwidths.append((template, variation, bandwidth))
        except RuntimeError as e:
            print(e.args[0], file=sys.stderr)
    
    optimal_badwidths.sort()
    
    
    with open(args.output, 'w') as out_file:
        out_file.write('#Template,Variation,h_angle,h_mass\n')
      
        for template, variation, bandwidth in optimal_badwidths:
            out_file.write('{},{},{:g},{:g}\n'.format(template, variation, *bandwidth))
    
    
    
    # Plot all bandwidths for selected templates
    mpl.rc('xtick', top=True, direction='in')
    mpl.rc('ytick', right=True, direction='in')
    mpl.rc('axes', labelsize='large')
    mpl.rc('axes.formatter', limits=[-3, 4], use_mathtext=True)
    mpl.rc('errorbar', capsize=2)
    mpl.rc('lines', markersize=4)
    
    if args.plots:
        templates_to_plot = args.plots.split(',')
    else:
        templates_to_plot = []
    
    
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
    
    
    sgn_name_regex = re.compile('gg([AH])_((pos|neg)-(sgn|int))-(.+)pc-M(\\d+)')
    sgn_part_labels = {'pos-sgn': 'R', 'pos-int': 'I^{+}', 'neg-int': 'I^{-}'}
    
    for template in templates_to_plot:
        match = sgn_name_regex.match(template)
        
        if not match:
            raise RuntimeError('Failed to parse signal template name "{}".'.format(template))
        
        sgn_label = 'CP-{}, $m = {:g}$ GeV, $\\Gamma / m = {:g}$%, ${}$'.format(
            'odd' if match.group(1) == 'A' else 'even',
            float(match.group(6)), float(match.group(5).replace('p', '.')),
            sgn_part_labels[match.group(2)]
        )
        
        
        for syst_write_name, syst_read_name in syst_names:
            
            errors = {}
            errors_raw = parse_file(
                os.path.join(args.inputs, '{}_{}.csv'.format(template, syst_read_name))
            )[2]
            
            for h_angle, h_mass, error in errors_raw:
                if h_angle in errors:
                    errors[h_angle][h_mass] = error
                else:
                    errors[h_angle] = {h_mass: error}
            
            
            fig = plt.figure()
            fig.patch.set_alpha(0.)
            axes = fig.add_subplot(111)
            
            for h_angle in sorted(errors.keys()):
                x, y = [], []
                
                for h_mass in sorted(errors[h_angle].keys()):
                    x.append(h_mass)
                    y.append(errors[h_angle][h_mass])
                
                axes.plot(
                    x, y, marker='o',
                    label=r'$h_\mathrm{{angle}} = {:g}$'.format(float(h_angle))
                )
            
            axes.legend()
            axes.set_xlabel(r'$h_\mathrm{mass}$')
            axes.set_ylabel(r'Mean $\chi^2$ error')
            
            axes.text(0., 1.005, sgn_label, ha='left', va='bottom', transform=axes.transAxes)
            axes.text(
                1., 1.005, syst_write_name,
                ha='right', va='bottom', transform=axes.transAxes
            )
            
            fig.savefig(
                os.path.join(args.fig_dir, '{}_{}_error.pdf'.format(template, syst_write_name))
            )
            plt.close(fig)

