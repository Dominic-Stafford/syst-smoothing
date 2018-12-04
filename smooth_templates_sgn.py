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

from smoothutils import AdaptiveRebinner, ReaderUnrolled, Smoother


if __name__ == '__main__':
    
    ROOT.gROOT.SetBatch(True)
    
    arg_parser = argparse.ArgumentParser(__doc__)
    arg_parser.add_argument('input', help='ROOT file with original templates')
    arg_parser.add_argument(
        '-b', '--bandwidths', default='bandwidths.csv',
        help='CSV file with chosen bandwidths'
    )
    arg_parser.add_argument(
        '-c', '--channel', default='ljets', help='Channels to process'
    )
    arg_parser.add_argument(
        '-o', '--output', default='templates.root',
        help='Name for output ROOT file with smoothed variations in signal'
    )
    args = arg_parser.parse_args()
    
    if args.channel == 'ljets':
        channels = ['mujets', 'ejets']
    elif args.channel == 'll':
        channels = ['ll']
    else:
        raise RuntimeError('Channel "{}" is not supported.'.format(args.channel))
    
    
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
    
    
    # Define variations to be smoothed.  Other templates will be copied
    # to the output file directly
    syst_names = []
    
    for jec_syst in [
        'AbsoluteStat', 'AbsoluteScale', 'AbsoluteMPFBias', 'Fragmentation',
        'SinglePionECAL', 'SinglePionHCAL', 'FlavorQCD', 'TimePtEta',
        'RelativeJEREC1', 'RelativePtBB', 'RelativePtEC1', 'RelativeBal', 'RelativeFSR',
        'RelativeStatFSR', 'RelativeStatEC',
        'PileUpDataMC', 'PileUpPtRef', 'PileUpPtBB', 'PileUpPtEC1'
    ]:
        syst_names.append('CMS_scale_j_13TeV_' + jec_syst)
    
    syst_names.append('CMS_res_j_13TeV')
    syst_names.append('CMS_METunclustered_13TeV')
    
    
    src_file = ROOT.TFile(args.input)
    out_file = ROOT.TFile(args.output, 'recreate')
    out_dirs = []
    
    templates_to_keep = []
    
    
    sgn_name_regex = re.compile(
        '(gg[AH]_(pos|neg)-(sgn|int)-.+pc-M\\d+)(_(.+)(Up|Down))?'
    )
    sgn_nominal_names = set()
    
    
    # Copy to the output file background templates, nominal signal
    # templates, and templates for systematic variations that do not
    # need to be smoothed
    for channel in channels:
        src_dir = src_file.Get(channel)
        out_dir = out_file.mkdir(src_dir.GetName())
        
        for template_key in src_dir.GetListOfKeys():
            template = template_key.ReadObj()
            
            if not template.InheritsFrom('TH1'):
                continue
            
            match = sgn_name_regex.match(template.GetName())
            
            if match:
                # Save name of the nominal signal template to be used
                # in smoothing
                sgn_nominal_names.add(match.group(1))
            
            if not match or match.group(5) not in syst_names:
                # This is either not a signal template or this is a
                # signal template but not a variation that needs to be
                # smoothed  (Group 5 in the regex is the name of the
                # variation.)  Copy the template to the output file with
                # no changes.
                template.SetDirectory(out_dir)
                templates_to_keep.append(template)
        
        out_dirs.append(out_dir)
    
    src_file.Close()
    
    
    # Proceed to smoothing of requested variations
    reader = ReaderUnrolled(args.input, num_bins_angle=5, channels=channels)
    
    for sgn_nominal_name in sgn_nominal_names:
        template_nominal = reader.read_counts(sgn_nominal_name)
        rebinner = AdaptiveRebinner(template_nominal)
        
        for syst_name in syst_names:
            
            # Optimal bandwidth for the current combination
            try:
                bandwidth = bandwidths[sgn_nominal_name, syst_name]
            except KeyError:
                raise KeyError('No bandwidth found for template "{}" variation "{}".'.format(
                    sgn_nominal_name, syst_name
                ))
            
            
            # Construct smoothed variations
            template_up = reader.read_counts('{}_{}Up'.format(sgn_nominal_name, syst_name))
            template_down = reader.read_counts('{}_{}Down'.format(sgn_nominal_name, syst_name))
            
            smoother = Smoother(
                template_nominal, template_up, template_down,
                rebinner, rebin_for_smoothing=True
            )
            template_up_smooth, template_down_smooth = smoother.smooth(
                (bandwidth[0] * reader.num_bins_angle, bandwidth[1] * reader.num_bins_mass)
            )
            
            
            # Add smoothed variations to output directory
            for ichannel, (direction, template_smooth) in itertools.product(
                range(len(channels)),
                [('up', template_up_smooth), ('down', template_down_smooth)]
            ):
                unrolled_template = np.reshape(template_smooth[ichannel], (-1, 2))
                hist = ROOT.TH1D(
                    '{}_{}{}'.format(sgn_nominal_name, syst_name, direction.capitalize()), '',
                    reader.num_bins_angle * reader.num_bins_mass,
                    0., reader.num_bins_angle * reader.num_bins_mass
                )
                
                for i in range(len(unrolled_template)):
                    hist.SetBinContent(i + 1, unrolled_template[i, 0])
                    hist.SetBinError(i + 1, math.sqrt(unrolled_template[i, 1]))
                
                hist.SetDirectory(out_dirs[ichannel])
                templates_to_keep.append(hist)
    
    
    for out_dir in out_dirs:
        out_dir.Write()
    
    out_file.Close()
    src_file.Close()
