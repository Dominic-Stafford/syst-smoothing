#!/usr/bin/env python

"""Produces smoothed templates for systematic variations.

Input variations are supposed to be a combination of all SM-driven
backgrounds.  The combined background is smoothed collectively and the
resulting variation is then attached to tt.
"""

import argparse
from collections import OrderedDict
import itertools
import json
import math
import os

import numpy as np

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from smoothutils import Reader, RebinnerND, Smoother


if __name__ == '__main__':
    
    argParser = argparse.ArgumentParser(epilog=__doc__)
    argParser.add_argument('inputFile', help='ROOT file histograms split into partitions')
    argParser.add_argument(
        '-b', '--bandwidths', default='bandwidths.json',
        help='Name of file with bandwidths to use'
    )
    argParser.add_argument(
        '-o', '--output', default='smoothTemplates.root',
        help='Name for output ROOT file with smoothed templates'
    )
    argParser.add_argument(
        '--fig-dir', dest='figDir', default='figSmooth',
        help='Name for directory to store figures'
    )
    args = argParser.parse_args()
    
    
    mpl.rc('xtick', top=True, direction='in')
    mpl.rc('ytick', right=True, direction='in')
    mpl.rc('axes', labelsize='large')
    mpl.rc('axes.formatter', limits=[-3, 4], use_mathtext=True)
    
    ROOT.gROOT.SetBatch(True)
    
    
    for subDir in ['', 'analysis', 'raw']:
        d = os.path.join(args.figDir, subDir)
        
        if not os.path.exists(d):
            os.makedirs(d)
    
    
    reader = Reader(args.inputFile)
    
    outFile = ROOT.TFile(args.output, 'recreate')
    channelDirs = {channel: outFile.mkdir(channel) for channel in reader.channels}
    histogramsToStore = []
    
    channelLabels = {'mujets': r'$\mu + \mathrm{jets}$', 'ejets': r'$e + \mathrm{jets}$'}
    
    
    # Construct an object to rebin NumPy arrays from underlying dense
    # binning to the coarser analysis-level binning
    sourceBinning, targetBinning = {}, {}
    
    with open('dense-binning.json') as f:
        binningInfo = json.load(f)
        sourceBinning['mass'] = binningInfo['mass']
        sourceBinning['angle'] = binningInfo['angle']
    
    with open('analysis-binning.json') as f:
        binningInfo = json.load(f)
        targetBinning['mass'] = binningInfo['MassTT']['binning']
        targetBinning['angle'] = binningInfo['CosTopLepTT']['binning']
    
    nBinsAngleAnalysis = len(targetBinning['angle']) - 1
    
    rebinner = RebinnerND([
        (2, sourceBinning['mass'], targetBinning['mass']),
        (1, sourceBinning['angle'], targetBinning['angle'])
    ])
    
    
    # Read nominal templates and produce a version rebinned for the
    # analysis binning.  Also read and rebin nominal templates for tt.
    nominalCounts = reader.read_counts_total('Nominal')
    nominalCountsRebinned = rebinner(nominalCounts)
    
    ttCountsRebinned = rebinner(reader.read_counts_total('TT'))
    
    nBinsAngle, nBinsMass = reader.nBinsAngle, reader.nBinsMass
    
    
    # Mapping between names of systematic uncertainties used by this
    # script and ones expected in combine data card
    systNameMatches = [
        ('FSR', 'QCDscaleFSR_TT'),
        ('TMass', 'TMass'),
        ('Hdamp', 'Hdamp_TT')
    ]
    
    for jecSyst in [
        'AbsoluteStat', 'AbsoluteScale', 'AbsoluteMPFBias', 'Fragmentation',
        'SinglePionECAL', 'SinglePionHCAL', 'FlavorQCD', 'TimePtEta',
        'RelativeJEREC1', 'RelativePtBB', 'RelativePtEC1', 'RelativeBal', 'RelativeFSR',
        'RelativeStatFSR', 'RelativeStatEC',
        'PileUpDataMC', 'PileUpPtRef', 'PileUpPtBB', 'PileUpPtEC1'
    ]:
        systNameMatches.append(('JEC' + jecSyst, 'CMS_scale_j_13TeV_{}'.format(jecSyst)))
    
    systNameMatches.extend([
        ('JER', 'CMS_res_j_13TeV'),
        ('METUncl', 'CMS_METunclustered_13TeV')
    ])
    
    
    
    # Optimal bandwidth for each uncertainty
    with open(args.bandwidths) as f:
        optimalBandwidths = json.load(f)
    
    
    # Systematic uncertainties that are defined with independent samples
    # (which should be taken into account when statistical uncertainties
    # are computed)
    systIndependentSamples = {'FSR', 'TMass', 'Hdamp'}
    
    
    for systName, systNameCard in systNameMatches:
        print('Uncertainty "{}"\n'.format(systName))
        
        
        # Read systematic variations
        systCounts = {
            direction: reader.read_counts_total(systName + direction.capitalize())
            for direction in ['up', 'down']
        }
        
        
        # Construct smoothed variations.  Optimal bandwidth is converted
        # from relative to absolute numbers, as expected by the
        # smoothing object.
        bandwidth = optimalBandwidths[systName]
        
        smoother = Smoother(
            nominalCounts, systCounts['up'], systCounts['down'], rebinner
        )
        systCountsSmoothRebinned = {}
        systCountsSmoothRebinned['up'], systCountsSmoothRebinned['down'] = \
            smoother.smooth(
                (bandwidth[0] * nBinsAngle, bandwidth[1] * nBinsMass)
            )
        
        # An alias for automatically constructed rebinned systematic
        # templates
        systCountsRebinned = {
            'up': smoother.systs_bin_sf[0],
            'down': smoother.systs_bin_sf[1]
        }
        
        print('  Scale factors fitted independently')
        print('    "up": {:+.2f} +- {:.2f}, "down": {:+.2f} +- {:.2f}'.format(
            *smoother.raw_scale_factors[0], *smoother.raw_scale_factors[1]
        ))
        print('  Final scale factors')
        print('    "up": {:+.2f}, "down": {:+.2f}'.format(*smoother.scale_factors))
        
        
        # Plot averaged smoothed deviations in each angular bin (with
        # the dense binning also in the angle)
        for binAngle in range(smoother.average_deviation.shape[0]):
            
            fig = plt.figure()
            axes = fig.add_subplot(111)
            
            axes.plot(smoother.average_deviation[binAngle], color='black', label='Original')
            axes.plot(smoother.smooth_average_deviation[binAngle], color='C1', label='Smoothed')
            
            axes.axhline(0., color='gray', lw=0.8, ls='dashed')
            axes.margins(x=0.)
            axes.legend()
            
            axes.set_xlabel('$m_{t\\bar t}$ bin index')
            axes.set_ylabel('Relative deviation from nominal')
            axes.text(
                0., 1.005,
                '{}, angle bin {}, $h = ({:g}, {:g})$'.format(systName, binAngle + 1, *bandwidth),
                ha='left', va='bottom', transform=axes.transAxes
            )
            
            fig.savefig(os.path.join(
                args.figDir, 'raw', '{}_smoothing_bin{}.pdf'.format(systName, binAngle + 1)
            ))
            plt.close(fig)
        
        
        # Compute chi^2 differences between inputs and smoothed rebinned
        # templates, as well as corresponding p-values
        compatibility = {}
        ndf = nominalCountsRebinned[0, ..., 0].size
        
        for iChannel, direction in itertools.product(
            range(nominalCounts.shape[0]), systCounts.keys()
        ):
            s = systCountsRebinned[direction][iChannel]
            sSmooth = systCountsSmoothRebinned[direction][iChannel]
            
            diff2 = (s[..., 0] - sSmooth[..., 0])**2
            
            # When systematic variations are described with independent
            # samples, sum in quadrature uncertainties from the nominal
            # template (propagated into sSmooth) and the original
            # variation.  When variations are constructed from the same
            # sample as the nominal template, use only the uncertainties
            # from s.
            unc2 = s[..., 1]
            
            if systName in systIndependentSamples:
                unc2 += sSmooth[..., 1]
            
            chi2 = np.sum(diff2 / unc2)
            compatibility[iChannel, direction] = chi2, ROOT.TMath.Prob(chi2, ndf)
        
        
        print()
        
        for iChannel, direction in itertools.product(
            range(nominalCounts.shape[0]), systCounts.keys()
        ):
            print('  Compatibility for variation "{}" channel "{}"'.format(
                direction, reader.channels[iChannel]
            ))
            print('    chi2/ndf: {0:5.2f} / {2}, p-value: {1:.3f}'.format(
                *compatibility[iChannel, direction], ndf
            ))
        
        
        # Plot relative deviations for input and smoothed templates.
        # For this 2D histograms of distributions of angle and mass are
        # unrolled.
        for iChannel, direction in itertools.product(
            range(nominalCounts.shape[0]), systCounts.keys()
        ):
            channel = reader.channels[iChannel]
            
            inputDeviation = systCountsRebinned[direction][iChannel, ..., 0] / \
                nominalCountsRebinned[iChannel, ..., 0] - 1
            smoothDeviation = systCountsSmoothRebinned[direction][iChannel, ..., 0] / \
                nominalCountsRebinned[iChannel, ..., 0] - 1
            
            inputDeviation = np.ravel(inputDeviation)
            smoothDeviation = np.ravel(smoothDeviation)
            
            
            fig = plt.figure()
            axes = fig.add_subplot(111)
            
            x = np.array(range(len(inputDeviation) + 1))
            axes.step(
                x, list(inputDeviation) + [0], where='post',
                solid_joinstyle='miter', color='black', label='Original'
            )
            axes.step(
                x, list(smoothDeviation) + [0], where='post',
                solid_joinstyle='miter', color='C1', label='Smoothed'
            )
            
            axes.axhline(0., color='gray', lw=0.8, ls='dashed')
            
            for i in range(nBinsAngleAnalysis - 1):
                axes.axvline(
                    len(inputDeviation) / nBinsAngleAnalysis * (i + 1),
                    color='gray', lw=0.8, ls='dashed'
                )
            
            axes.margins(x=0.)
            axes.legend()
            
            axes.set_xlabel(r'$m_{t\bar t} \otimes \cos\/\theta^*_{\mathrm{t}_\ell}$ bin index')
            axes.set_ylabel('Relative deviation from nominal')
            axes.text(
                0., 1.005,
                '{} ({}), {}, $h = ({:g}, {:g})$, $\\chi^2 = {:.2f}$'.format(
                    systName, direction, channelLabels[channel], *bandwidth,
                    compatibility[iChannel, direction][0]
                ),
                ha='left', va='bottom', transform=axes.transAxes
            )
            
            fig.savefig(os.path.join(
                args.figDir, 'analysis',
                '{}_{}_{}.pdf'.format(systName, channel, direction.lower())
            ))
            plt.close(fig)
        
        
        # Finally, store smoothed templates in output file
        for iChannel, channel in enumerate(reader.channels):
            for direction in ['up', 'down']:
                
                # Take the absolute smoothed deviation, as computed with
                # all processes, and apply it to tt alone
                template = ttCountsRebinned[iChannel] + \
                    systCountsSmoothRebinned[direction][iChannel] - nominalCountsRebinned[iChannel]
                
                template = np.reshape(template, (-1, 2))
                
                
                histogram = ROOT.TH1D(
                    'TT_{}{}'.format(systNameCard, direction.capitalize()), '',
                    len(template), 0., len(template)
                )
                histogram.SetDirectory(channelDirs[channel])
                
                for i in range(template.shape[0]):
                    histogram.SetBinContent(i + 1, template[i, 0])
                    histogram.SetBinError(i + 1, math.sqrt(template[i, 1]))
                
                # Prevent garbage collector from deleting the histogram
                histogramsToStore.append(histogram)
        
        
        # Separation between outputs for different uncertainties
        print('\n')
    
    
    outFile.Write()
