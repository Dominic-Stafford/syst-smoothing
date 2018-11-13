#!/usr/bin/env python

"""Runs cross-validation to determine optimal bandwidth for smoothing.

This script runs repeated cross-validation and saves smoothing errors
obtained at each iteration in a text file.  New results are appended to
the file.  A random seed is used to shuffle data for cross-validation,
which means that results from different runs can be combined.
"""

import argparse
from collections import defaultdict, OrderedDict
import itertools
import json
import math
import multiprocessing as mp
import queue

import numpy as np

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from smoothutils import Reader, RebinnerND, RepeatedCV, Smoother


class Processor(mp.Process):
    """A class to run cross-validation in multiple processes.
    
    This class defines a single process, and multiple instances of it
    are meant to run in parallel.  The work is splitted at the level of
    individual uncertainties, whereas for each individual uncertainty
    all processing is performed by the same instance.
    """
    
    def __init__(self, inputFileName, kCV, numRepeat, inputs, results):
        """Constructor.
        
        Arguments:
            inputFileName:  Name of ROOT file with histograms.
            kCV:  Desired number of "folds" for cross-validation.
            numRepeat:  Desired number of times to repeat
                cross-valiadation, having shuffled data.
            inputs:  Input queue with names of uncertainties.
            results:  Output queue.
        """
        
        super().__init__()
        
        self.inputs = inputs
        self.results = results
        
        
        # Create an object to construct test and training sets for CV
        self.reader = Reader(inputFileName)
        self.repeatedCV = RepeatedCV(self.reader, kCV)
        self.numRepeat = numRepeat
        
        
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
        
        self.rebinner = RebinnerND([
            (2, sourceBinning['mass'], targetBinning['mass']),
            (1, sourceBinning['angle'], targetBinning['angle'])
        ])
        
        
        # Values for bandwidths along each axis to be tested
        self.bandwidths = (
            [0.2, 0.5, 1.],  # Angle
            [0.05, 0.1, 0.2, 0.3, 0.5, 1.]  # Mass
        )
    
    
    def run(self):
        """Read uncertainties from input queue and run cross-validation.
        
        The result for each uncertainty and bandwidth is a list of
        errors obtained in every CV iteration and every repetition of
        the CV procedure.  It is put into the output queue.
        """
        
        while True:
            try:
                systName = self.inputs.get_nowait()
            except queue.Empty:
                # Put a "poison pill" into the output queue so that the
                # main process can figure out when all results have been
                # produced.  This is probably not the most elegant
                # solution though.
                self.results.put(None)
                break
            
            
            self.repeatedCV.book(['Nominal', systName + 'Up', systName + 'Down'])
            cvErrors = defaultdict(list)
            
            for iRepetition in range(self.numRepeat):
                self.repeatedCV.shuffle()
                
                for iCV in range(self.repeatedCV.kCV):
                    self.repeatedCV.create_cv_partitions(iCV)
                    
                    # Rebin counts needed to evaluate smoothing error
                    nominalTrainRebinned = self.rebinner(
                        self.repeatedCV.get_counts_train('Nominal')
                    )
                    nominalTestRebinned = self.rebinner(self.repeatedCV.get_counts_test('Nominal'))
                    systTestRebinned = (
                        self.rebinner(self.repeatedCV.get_counts_test(systName + 'Up')),
                        self.rebinner(self.repeatedCV.get_counts_test(systName + 'Down'))
                    )
                    
                    
                    # An object to perform smoothing
                    smoother = Smoother(
                        self.repeatedCV.get_counts_train('Nominal'),
                        self.repeatedCV.get_counts_train(systName + 'Up'),
                        self.repeatedCV.get_counts_train(systName + 'Down'),
                        self.rebinner, nominalRebinned=nominalTrainRebinned
                    )
                    
                    # Perform smoothing and compute error for each
                    # bandwidth.  Smoothing algorithm expects the bandwidth
                    # in absolute numbers.
                    for bandwidth in itertools.product(*self.bandwidths):
                        smoothUpRebinned, smoothDownRebinned = smoother.smooth(
                            (
                                bandwidth[0] * self.reader.nBinsAngle,
                                bandwidth[1] * self.reader.nBinsMass
                            ),
                            rebin=True
                        )
                        error = self.compute_error(
                            nominalTrainRebinned, (smoothUpRebinned, smoothDownRebinned),
                            nominalTestRebinned, systTestRebinned
                        )
                        cvErrors[bandwidth].append(error)
            
            
            for bandwidth, values in cvErrors.items():
                self.results.put((systName, bandwidth, values))
            
    
    @staticmethod
    def compute_error(nominalTrain, systSmooth, nominalTest, systTest):
        """Compute smoothing error.
        
        The error is defined as the chi^2 deviation between the smoothed
        relative deviation and the actual deviation on the test partition.
        The uncertainty for chi^2 is computed from uncertainties in the
        nominal and systematic templates as if they were indepenend.  This
        is not true for uncertainties like JEC but still acceptable since
        the error is only used to rank different values for bandwidths.
        """
        
        chi2 = 0.
        
        # Sum chi^2 from up and down variations
        for i in range(2):
            
            # Squared difference between relative deviations
            diff2 = (systTest[i][..., 0] / nominalTest[..., 0] - \
                systSmooth[i][..., 0] / nominalTrain[..., 0])**2
            
            # Squared uncertainty computed with the usual error
            # propagation:
            #  Var[s/n - 1] = Var[s] / n^2 + Var[n] * s^2 / n^4
            unc2 = systTest[i][..., 1] / nominalTest[..., 0]**2 + \
                nominalTest[..., 1] * systTest[i][..., 0]**2 / nominalTest[..., 0]**4
            
            chi2 += np.sum(diff2 / unc2)
        
        return chi2


if __name__ == '__main__':
    
    argParser = argparse.ArgumentParser(epilog=__doc__)
    argParser.add_argument('inputFile', help='ROOT file histograms split into partitions')
    argParser.add_argument(
        '-k', '--k-cv', dest='kCV', type=int, default=10,
        help='Number of folds for cross-validation'
    )
    argParser.add_argument(
        '-r', '--repeat', type=int, default=100,
        help='Number of times to repeat cross-validation'
    )
    argParser.add_argument(
        '-p', '--parallel', type=int, default=16,
        help='Number of processes to run in parallel'
    )
    argParser.add_argument(
        '-o', '--output', default='crossValidation.data',
        help='Name for output file with CV errors'
    )
    args = argParser.parse_args()
    
    ROOT.gROOT.SetBatch(True)
    
    
    # Create input queue with names of uncertainties to process and
    # output queue to collect results
    reader = Reader(args.inputFile)
    systNames = reader.get_syst_names()
    
    systNamesQueue = mp.Queue()
    
    for systName in systNames:
        systNamesQueue.put(systName)
    
    results = []
    resultQueue = mp.Queue()
    
    
    # Run cross-validation in a process pool
    processors = []
    
    for i in range(args.parallel):
        p = Processor(args.inputFile, args.kCV, args.repeat, systNamesQueue, resultQueue)
        processors.append(p)
        p.start()
    
    
    blocksToGet = len(processors)
    
    while blocksToGet:
        res = resultQueue.get()
        
        if res is None:
            blocksToGet -= 1
        else:
            results.append(res)
    
    
    for p in processors:
        p.join()
    
    
    # Append new results to given output file.  Its content is not
    # overwritten.
    with open(args.output, 'a') as f:
        for result in results:
            f.write('{}  {:.2f}  {:.2f} '.format(result[0], result[1][0], result[1][1]))
            
            for error in result[2]:
                f.write(' {}'.format(error))
            
            f.write('\n')
