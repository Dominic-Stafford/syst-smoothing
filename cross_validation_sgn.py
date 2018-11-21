#!/usr/bin/env python

"""Runs cross-validation to determine optimal bandwidth for smoothing.

This script runs repeated cross-validation and saves in a text file mean
smoothing errors computed from all iterations.  Values of the bandwidths
to probe are hard-coded.
"""

import argparse
import itertools

import numpy as np

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from smoothutils import AdaptiveRebinner, ReaderCV, RepeatedCV, Smoother


def compute_error(nominal_train, syst_smooth, nominal_test, syst_test, max_rel_unc=0.5):
    """Compute smoothing error.
    
    The error is defined as the chi^2 deviation between the smoothed
    relative deviation and the actual deviation on the test partition.
    The uncertainty for chi^2 is computed from uncertainties in the
    nominal and systematic templates as if they were indepenend.  This
    is not true for uncertainties like JEC but still acceptable since
    the error is only used to rank different values for bandwidths.
    Underpopulated bins, as defined by the maximal relative uncertainty,
    are skipped from the computation.
    """
    
    # Identify properly populated bins.  For empty bins the ratio
    # evaluates to NaN, and the result of any comparison of it with a
    # number is False, so these bins will also be excluded.
    with np.errstate(all='ignore'):
        populated_bins = np.logical_and(
            np.sqrt(nominal_test[..., 1]) / nominal_test[..., 0] < max_rel_unc,
            np.sqrt(nominal_train[..., 1]) / nominal_train[..., 0] < max_rel_unc
        )
    
    
    chi2 = 0.
    
    # Sum chi^2 from up and down variations
    for i in range(2):
        with np.errstate(all='ignore'):
            # Squared difference between relative deviations
            diff2 = (syst_test[i][..., 0] / nominal_test[..., 0] - \
                syst_smooth[i][..., 0] / nominal_train[..., 0]) ** 2
            
            # Squared uncertainty computed with the usual error
            # propagation:
            #  Var[s/n - 1] = Var[s] / n^2 + Var[n] * s^2 / n^4
            # Include terms for both test and training set because the
            # uncertainty for computed on the test set alone can be
            # grossly wrong when the number of events in it is small.
            # There is some double counting of the uncertainty, but its
            # absolute scale is not important to choose the bandwidth.
            unc2 = syst_test[i][..., 1] / nominal_test[..., 0] ** 2 + \
                nominal_test[..., 1] * syst_test[i][..., 0] ** 2 / nominal_test[..., 0] ** 4 + \
                syst_smooth[i][..., 1] / nominal_train[..., 0] ** 2 + \
                nominal_train[..., 1] * syst_smooth[i][..., 0] ** 2 / nominal_train[..., 0] ** 4
            
            r2 = diff2 / unc2
        
        chi2 += np.sum(r2[populated_bins])
    
    
    # Sanity checks
    if not np.isfinite(chi2):
        raise RuntimeError('Obtained chi^2 is not a finite number.')
    
    if chi2 == 0:
        raise RuntimeError('Obtained a zero chi^2.')
    
    return chi2


if __name__ == '__main__':
    
    ROOT.gROOT.SetBatch(True)
    
    arg_parser = argparse.ArgumentParser(__doc__)
    arg_parser.add_argument('file', help='ROOT file histograms split into partitions')
    arg_parser.add_argument('template', help='Name for nominal signal template')
    arg_parser.add_argument('variation', help='Name for variation')
    arg_parser.add_argument(
        '-k', '--k-cv', dest='k_cv', type=int, default=10,
        help='Number of folds for cross-validation'
    )
    arg_parser.add_argument(
        '-r', '--repeat', type=int, default=100,
        help='Number of times to repeat cross-validation'
    )
    arg_parser.add_argument(
        '--max-unc', type=float, default=0.05,
        help='Maximal relative uncertainty for adaptive rebinning'
    )
    arg_parser.add_argument(
        '-o', '--output', default='scores.csv',
        help='Name for output file with CV error'
    )
    args = arg_parser.parse_args()
    
    # Bandwidths to try
    bandwidths = list(itertools.product(
        [0.3, 0.5, 1., 1.5],      # Angle
        [0.2, 0.3, 0.5, 1., 1.5]  # Mass
    ))
    
    
    reader = ReaderCV(args.file)
    rebinner = AdaptiveRebinner(reader.read_counts(args.template))
    
    nominal_name = args.template
    up_name = '{}_{}Up'.format(args.template, args.variation)
    down_name = '{}_{}Down'.format(args.template, args.variation)
    
    repeated_cv = RepeatedCV(reader, args.k_cv)
    repeated_cv.book([nominal_name, up_name, down_name])
    
    cv_errors = {bandwidth: [] for bandwidth in bandwidths}
    
    for repetition in range(args.repeat):
        repeated_cv.shuffle()
        
        for icv in range(repeated_cv.k_cv):
            repeated_cv.create_cv_partitions(icv)
            
            nominal_test_rebinned = rebinner(repeated_cv.get_counts_test(nominal_name))
            up_test_rebinned = rebinner(repeated_cv.get_counts_test(up_name))
            down_test_rebinned = rebinner(repeated_cv.get_counts_test(down_name))
            
            smoother = Smoother(
                repeated_cv.get_counts_train(nominal_name),
                repeated_cv.get_counts_train(up_name), repeated_cv.get_counts_train(down_name),
                rebinner, rebin_for_smoothing=True
            )
            
            for bandwidth in bandwidths:
                up_smooth, down_smooth = smoother.smooth((
                    bandwidth[0] * reader.num_bins_angle,
                    bandwidth[1] * reader.num_bins_mass
                ))
                
                # Compute approximation error using the coarse binning
                error = compute_error(
                    smoother.nominal,
                    (rebinner(up_smooth), rebinner(down_smooth)),
                    nominal_test_rebinned, (up_test_rebinned, down_test_rebinned)
                )
                cv_errors[bandwidth].append(error)
    
    
    # Write results to a file
    with open(args.output, 'w') as f:
        f.write('#Hypothesis,Variation,h_angle,h_mass,N,Mean chi2,Std chi2\n')
        for bandwidth in bandwidths:
            errors = cv_errors[bandwidth]
            f.write('{},{},{:g},{:g},{:d},{:g},{:g}\n'.format(
                args.template, args.variation, bandwidth[0], bandwidth[1],
                len(errors), np.mean(errors), np.std(errors)
            ))
