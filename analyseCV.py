#!/usr/bin/env python

"""Analyses output files produced by crossValidation.py.

Plots mean CV errors for all probed bandwidths and, for each
uncertainty, chooses the bandwidth that gives the smallest CV error.
However, the latter is done under a condition that the bandwidth is
larger than a threshold.
"""

import argparse
from collections import OrderedDict
import json
import math
import os

import numpy as np

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt


if __name__ == '__main__':
    
    argParser = argparse.ArgumentParser(epilog=__doc__)
    argParser.add_argument('inputFile', help='File produced by crossValidation.py')
    argParser.add_argument(
        '-o', '--output', default='bandwidths.json',
        help='Name for output file with chosen bandwidths'
    )
    argParser.add_argument(
        '--fig-dir', dest='figDir', default='figCV',
        help='Directory for produced figures'
    )
    args = argParser.parse_args()
    
    if not os.path.exists(args.figDir):
        os.makedirs(args.figDir)
    
    
    mpl.rc('xtick', top=True, direction='in')
    mpl.rc('ytick', right=True, direction='in')
    mpl.rc('axes', labelsize='large')
    mpl.rc('axes.formatter', limits=[-3, 4], use_mathtext=True)
    mpl.rc('errorbar', capsize=2)
    mpl.rc('lines', markersize=4)
    
    
    # Read smoothing errors from the input file.  Organize them by names
    # of systematic uncertainties and by bandwidths.
    smoothingErrors = {}
    
    with open(args.inputFile) as inputFile:
        for line in inputFile:
            tokens = line.split()
            
            systName = tokens[0]
            bandwidth = (tokens[1], tokens[2])  # Keep as strings
            errors = [float(t) for t in tokens[3:]]
            
            if systName not in smoothingErrors:
                smoothingErrors[systName] = {}
            
            d = smoothingErrors[systName]
            
            if bandwidth not in d:
                d[bandwidth] = []
            
            d = d[bandwidth]
            d.extend(errors)
    
    
    optimalBandwidths = OrderedDict()
    minimalMassBandwidth = 0.09
    
    for systName in sorted(smoothingErrors):
        
        # Compute mean errors and their standard deviations
        stats = {}
        
        for bandwidth, errors in smoothingErrors[systName].items():
            stats[bandwidth] = (len(errors), np.mean(errors), np.std(errors, ddof=1))
        
        
        # Find bandwidth that gives the smallest error
        minError = float('inf')
        optimalBandwidth = None
        
        for bandwidth, stat in stats.items():
            b = (float(bandwidth[0]), float(bandwidth[1]))
            
            # Do not consider mass bandwidths smaller than the threshold
            if b[1] < minimalMassBandwidth:
                continue
            
            if stat[1] < minError:
                minError = stat[1]
                optimalBandwidth = b
        
        optimalBandwidths[systName] = optimalBandwidth
        
        
        # Build a correspondence between bandwidths along angle and mass
        # axes
        bandwidthMap = {}
        
        for bandwidth in smoothingErrors[systName]:
            if bandwidth[0] not in bandwidthMap:
                bandwidthMap[bandwidth[0]] = set()
            
            bandwidthMap[bandwidth[0]].add(bandwidth[1])
        
        
        # Plot mean errors
        fig = plt.figure()
        axes = fig.add_subplot(111)
        
        for bandwidthAngle in sorted(bandwidthMap.keys(), key=lambda x: float(x)):
            x, y, yErr = [], [], []
            
            for bandwidthMass in sorted(bandwidthMap[bandwidthAngle], key=lambda x: float(x)):
                stat = stats[bandwidthAngle, bandwidthMass]
                
                x.append(float(bandwidthMass))
                y.append(stat[1])
                yErr.append(stat[2] / math.sqrt(stat[0]))
            
            # axes.errorbar(
            #     x, y, yerr=yErr,
            #     marker='o',
            #     label=r'$h_\mathrm{{angle}} = {:g}$'.format(float(bandwidthAngle))
            # )
            axes.plot(
                x, y, marker='o',
                label=r'$h_\mathrm{{angle}} = {:g}$'.format(float(bandwidthAngle))
            )
        
        axes.legend()
        axes.set_xlabel(r'$h_\mathrm{mass}$')
        axes.set_ylabel(r'Mean $\chi^2$ error')
        axes.text(
            0., 1.005, systName,
            ha='left', va='bottom', transform=axes.transAxes
        )
        
        fig.savefig(os.path.join(args.figDir, '{}_meanError.pdf'.format(systName)))
        plt.close(fig)
        
        
        # Plot standard deviations
        fig = plt.figure()
        axes = fig.add_subplot(111)
        
        for bandwidthAngle in sorted(bandwidthMap.keys(), key=lambda x: float(x)):
            x, y = [], []
            
            for bandwidthMass in sorted(bandwidthMap[bandwidthAngle], key=lambda x: float(x)):
                stat = stats[bandwidthAngle, bandwidthMass]
                
                x.append(float(bandwidthMass))
                y.append(stat[2] / stat[1] * 100.)
            
            axes.plot(
                x, y, marker='o',
                label=r'$h_\mathrm{{angle}} = {:g}$'.format(float(bandwidthAngle))
            )
        
        axes.legend()
        axes.set_xlabel(r'$h_\mathrm{mass}$')
        axes.set_ylabel(r'Standard deviaton of $\chi^2$ error [%]')
        axes.text(
            0., 1.005, systName,
            ha='left', va='bottom', transform=axes.transAxes
        )
        
        fig.savefig(os.path.join(args.figDir, '{}_std.pdf'.format(systName)))
        plt.close(fig)
    
    
    with open(args.output, 'w') as f:
        json.dump(optimalBandwidths, f, indent=2)
