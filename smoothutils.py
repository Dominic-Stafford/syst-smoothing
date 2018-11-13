"""Provides tools for smoothing of systematic variations."""

import itertools
import math
from uuid import uuid4

import numpy as np

import ROOT

from lowess import lowess, lowess2d, lowess2d_grid


class Reader:
    """Facilitates reading of files with histograms.
    
    Histograms are supposed to be organized into directories
    corresponding to channels.
    """
    
    def __init__(self, fileName):
        """Constructor from a file name."""
        
        self.templateFile = ROOT.TFile(fileName)
        
        
        # Extract the list of channels
        self.channels = []
        
        for key in self.templateFile.GetListOfKeys():
            if key.GetClassName() == 'TDirectoryFile':
                self.channels.append(key.GetName())
        
        
        # Extract the number of bins in histograms.  Since they all have
        # the same shape, take one of the nominal histograms.  Count
        # under- and overflow bins for the mass but not for the angle.
        hist = self.templateFile.Get(self.channels[0] + '/Nominal')
        self.nBinsMass = hist.GetNbinsX() + 2
        self.nBinsAngle = hist.GetNbinsY()
        self.nPartitions = hist.GetNbinsZ()
    
    
    def get_syst_names(self):
        """Provide list of systematic uncertainties included in file."""
        
        # Assume that the list of uncertainties is the same for all
        # channels
        systNames = []
        
        for key in self.templateFile.Get(self.channels[0]).GetListOfKeys():
            name = key.GetName()
            
            if name.endswith('Up'):
                systNames.append(name[:-2])
        
        return systNames
    
    
    def read_counts_partitions(self, name, partitions):
        """Read partitions from set of histograms.
        
        Read a set of histograms with the given name and merge requested
        partitions.  Return a NumPy representation of the resulting
        histogram.
        
        Arguments:
            name:  Name that identifies the set of histograms.
            partitions:  List of zero-based indices of partitions to be
                read.  The largest value must be smaller than
                self.nPartitions.
        
        Return value:
            NumPy array describing the histogram.  Its indices have the
            following meaning: (0) channel, (1) bin in angle, (2) bin in
            mass, (3) index distinguishing between content of bin of
            original histogram or its squared uncertainty.
        """
        
        counts = np.zeros((len(self.channels), self.nBinsAngle, self.nBinsMass, 2))
        
        for iChannel, channel in enumerate(self.channels):
            histName = '{}/{}'.format(channel, name)
            hist = self.templateFile.Get(histName)
            
            if not hist:
                raise RuntimeError('Failed to read histogram "{}".'.format(histName))
            
            for partition, binMass, binAngle in itertools.product(
                partitions, range(hist.GetNbinsX() + 2), range(1, hist.GetNbinsY() + 1)
            ):
                counts[iChannel, binAngle - 1, binMass, 0] += \
                    hist.GetBinContent(binMass, binAngle, partition + 1)
                counts[iChannel, binAngle - 1, binMass, 1] += \
                    hist.GetBinError(binMass, binAngle, partition + 1)**2
        
        return counts
    
    
    def read_counts_total(self, name):
        """Read set of histograms with given name.
        
        Same as read_counts but all partitions are merged together.
        """
        
        counts = np.empty((len(self.channels), self.nBinsAngle, self.nBinsMass, 2))
        
        for iChannel, channel in enumerate(self.channels):
            histName = '{}/{}'.format(channel, name)
            hist = self.templateFile.Get(histName)
            
            if not hist:
                raise RuntimeError('Failed to read histogram "{}".'.format(histName))
            
            hist.GetZaxis().SetRange(1, self.nPartitions)
            hist2D = hist.Project3D('yx e')
            hist2D.SetName(uuid4().hex)
            hist2D.SetDirectory(None)
            
            counts[iChannel] = self._hist_to_array(hist2D)
        
        return counts
    
    
    def _hist_to_array(self, hist):
        """Convert histogram of (mass, angle) into NumPy representation.
        
        Return an array of shape (nBinsAngle, nBinsMass, 2), where the
        last dimension includes bin contents and respective squared
        errors.
        """
        
        counts = np.empty((self.nBinsAngle, self.nBinsMass, 2))
        
        for binAngle, binMass in itertools.product(
            range(1, hist.GetNbinsY() + 1), range(hist.GetNbinsX() + 2)
        ):
            counts[binAngle - 1, binMass, 0] = hist.GetBinContent(binMass, binAngle)
            counts[binAngle - 1, binMass, 1] = hist.GetBinError(binMass, binAngle)**2
        
        return counts



class Rebinner1D:
    """A class to rebin NumPy arrays along an axis."""
    
    def __init__(self, sourceBinning, targetBinning, axis=None):
        """Constructor from source and target binnings.
        
        The binnings must satisfy certain requirements listed for
        _build_rebin_map.
        """
        
        self.rebinMap = self._build_rebin_map(sourceBinning, targetBinning)
        self.defaultAxis = axis
    
    
    def __call__(self, array, axis=None):
        """Create a new array with target binning along given axis."""
        
        if axis is None:
            axis = self.defaultAxis
        
        if axis is None:
            raise RuntimeError('No axis provided')
        
        
        rebinnedShape = list(array.shape)
        rebinnedShape[axis] = len(self.rebinMap) - 1
        
        rebinnedArray = np.zeros(rebinnedShape, dtype=array.dtype)
        
        for i in range(len(self.rebinMap) - 1):
            
            # An indexing object that selects a slice of rebinnedArray
            # at index i along the given axes
            index = tuple([slice(None)] * axis + [i, ...])
            
            rebinnedArray[index] = np.sum(
                np.take(array, range(self.rebinMap[i], self.rebinMap[i + 1]), axis=axis),
                axis=axis
            )
        
        return rebinnedArray
    
    
    @staticmethod
    def _build_rebin_map(sourceBinning, targetBinning):
        """Construct mapping between source and target binning.
        
        Assume that all elements of the target binning are included into
        the source binning and both arrays are sorted.  Also assume that
        first and last bins from the source binning are included into
        the respective bins of the target binning.  Return a list such
        that element number i in it is the (zero-based) bin number of
        the first bin from the source binning included in bin i of the
        target binning.
        """
        
        sourceBinning = np.asarray(sourceBinning)
        targetBinning = np.asarray(targetBinning)
        
        
        # Use half of the width of the narrowest bin in the source
        # binning to match the binnings
        eps = np.min(sourceBinning[1:] - sourceBinning[:-1]) / 2
        
        startIndicesTarget = [0]
        
        for edge in targetBinning[1:-1]:
            startIndicesTarget.append(np.searchsorted(sourceBinning, edge + eps) - 1)
        
        # The last element is the number of bins in the source binning
        # (which is one smaller than the number of edges)
        startIndicesTarget.append(len(sourceBinning) - 1)
        
        return startIndicesTarget



class RebinnerND:
    """A class to rebin NumPy arrays along multiple axes.
    
    Performs the rebinning by consequently applying instances of
    Rebinner1D along each axis.
    """
    
    def __init__(self, binnings):
        """Construct from source and target binnings.
        
        The argument is interpreted as a list of tuples (axis, source,
        target), where 'source' and 'target' and source and target
        binnings along axis 'axis'.
        """
        
        # Construct 1D rebinning objects
        self.rebinners = [
            Rebinner1D(b[1], b[2], axis=b[0]) for b in binnings
        ]
    
    
    def __call__(self, array):
        """Create a new rebinned array."""
        
        rebinnedArray = array
        
        for rebinner in self.rebinners:
            rebinnedArray = rebinner(rebinnedArray)
        
        return rebinnedArray



class RepeatedCV:
    """A class to provide inputs for repeated cross-validation.
    
    This class automatizes creation of test and training sets for
    cross-validation (CV).  The whole CV procedure can be repeated
    multiple times after reshuffling data.
    """
    
    def __init__(self, reader, kCV):
        """Constructor from a reader object and number of CV partitions.
        
        Arguments:
            reader:  An instance of class Reader
            kCV:  Desired number of CV partitions.  Should normally be
                a divisor of the number of raw partitions in the reader.
        """
        
        self.reader = reader
        self.kCV = kCV
        
        # Booked total counts
        self.totalCounts = {}
        
        # Order in which partitions will be iterated during
        # cross-validation
        self.partitions = list(range(reader.nPartitions))
        
        # Current partitions that constituent current test set and
        # corresponding counts for booked histograms
        self.testPartitions = None
        self.testCounts = {}
        
        # Complementary counts that define the training set
        self.trainCounts = {}
    
    
    def book(self, names):
        """Specify names of histogram sets that will be used."""
        
        newTotalCounts = {}
        
        for name in names:
            if name in self.totalCounts:
                # Already booked.  No need to read again.
                newTotalCounts[name] = self.totalCounts[name]
            else:
                newTotalCounts[name] = self.reader.read_counts_total(name)
        
        self.totalCounts = newTotalCounts
    
    
    def create_cv_partitions(self, k):
        """Create CV partitions.
        
        Define test and training partitions.
        
        Arguments:
            k:  Zero-based index of test CV partition.  Remaining CV
                partitions define the training data.
        """
        
        if k < 0 or k >= self.kCV:
            raise RuntimeError('Illegal index.')
        
        cvLength = round(len(self.partitions) / self.kCV)
        
        if k < self.kCV - 1:
            self.testPartitions = self.partitions[k * cvLength : (k + 1) * cvLength]
        else:
            # Special treatment for the last CV partition in case the
            # number of CV folds is not aligned with the number of
            # partitions.  This way all data are utilized.
            self.testPartitions = self.partitions[k * cvLength :]
        
        
        # Update counts for booked histograms
        self.testCounts, self.trainCounts = {}, {}
        
        for name, totalCounts in self.totalCounts.items():
            testCounts = self.reader.read_counts_partitions(name, self.testPartitions)
            self.testCounts[name] = testCounts
            self.trainCounts[name] = totalCounts - testCounts
    
    
    def get_counts_test(self, name):
        """Provide counts in the current test set."""
        
        if not self.testCounts:
            raise RuntimeError('CV partitions have not been created.')
        
        if name not in self.testCounts:
            raise RuntimeError('Counts with name "{}" have not been booked.'.format(name))
        
        return self.testCounts[name]
    
    
    def get_counts_train(self, name):
        """Provide counts in the current training set."""
        
        if not self.trainCounts:
            raise RuntimeError('CV partitions have not been created.')
        
        if name not in self.trainCounts:
            raise RuntimeError('Counts with name "{}" have not been booked.'.format(name))
        
        return self.trainCounts[name]
    
    
    def shuffle(self):
        """Shuffle underlying raw partitions.
        
        This allows to repeat the full CV procedure anew.
        """
        
        np.random.shuffle(self.partitions)
        self.testPartitions = None
        self.testCounts = {}



class Smoother:
    """A class to smooth systematic variations."""
    
    def __init__(
        self, nominal, up, down, rebinner,
        nominalRebinned=None, upRebinned=None, downRebinned=None,
        sfCombThreshold=1.
    ):
        """Constructor from reference templates.
        
        Arguments:
            nominal, up, down:  Reference templates with fine binning.
                They must respect the format adopted in Reader, i.e.
                their shapes must be (nChannels, nBinsAngle, nBinsMass,
                2).
            rebinner:  An instance of RebinnerND
            nominalRebinned, upRebinned, downRebinned:  Reference
                templates rebinned for the analysis binning.  Any of
                these arguments can be None.  In this case rebinned
                template is constructed automatically.
            sfCombThreshold:  If the difference between scale factors
                for up and down (with flipped sign) variations computed
                independently is smaller than their combined uncertainty
                multiplied by the given factor, will use the same scale
                factor for both (up to the sign).
        
        The rebinner is needed to compute scale factors using a coarser
        binning.
        """
        
        self.nominal = nominal
        self.systs = (up, down)
        self.rebinner = rebinner
        
        # Precompute relative deviation from the nominal template,
        # averaged over channels and up and down variations.  The sign
        # is chosen in such a way that obtained deviation corresponds to
        # the up variation.  The shape of produced array is
        # (nBinsAngle, nBinsMass).
        self.averageDeviation = (np.sum(up[..., 0], axis=0) - np.sum(down[..., 0], axis=0)) / 2 / \
            np.sum(nominal[..., 0], axis=0)
        
        self.smoothAverageDeviation = None
        
        
        # Precompute rebinned templates if needed
        if nominalRebinned is None:
            self.nominalRebinned = rebinner(nominal)
        else:
            self.nominalRebinned = nominalRebinned
        
        if upRebinned is None:
            upRebinned = rebinner(up)
        
        if downRebinned is None:
            downRebinned = rebinner(down)
        
        self.systsRebinned = (upRebinned, downRebinned)
        
        
        self.sfCombThreshold = sfCombThreshold
        self.scaleFactors, self.rawScaleFactors = None, None
            

    def smooth(self, bandwidth, algo='2D', rebin=False):
        """Perform LOWESS smoothing using given bandwidth.
        
        Arguments:
            bandwidth:  Bandwidth used for smoothing.  Exact meaining
                depends on the algorithm, see documentation for
                respective methods.
            algo:  Smoothing algorithm.  Supported are '2D' and 'mass'.
            rebin:  Controls whether returned templates use the fine
                or analysis-level binning.
        
        Return value:
            A pair of templates that represent up and down smoothed
            variations.  Their shapes are (nChannels, nBinsAngle,
            nBinsMass, 2).  The uncertainty is computed from the
            uncertainty of the nominal template.
        
        Add to 'self' smoothed averaged relative deviation and scale
        factors for up and down deviations.
        """
        
        # Construct smoothed average deviation
        if algo == '2D':
            self.smoothAverageDeviation = self._smooth_impl_2d(bandwidth)
        elif algo == 'mass':
            self.smoothAverageDeviation = self._smooth_impl_mass(bandwidth)
        else:
            raise RuntimeError('Unknown algorithm "{}".'.format(algo))
        
        
        # Compute scale factors for up and down variations
        self._compute_scale_factors()
        
        
        # Construct smoothed systematic templates
        smoothTemplates = (
            self._apply_deviation(self.smoothAverageDeviation, scaleFactor=self.scaleFactors[0]),
            self._apply_deviation(self.smoothAverageDeviation, scaleFactor=self.scaleFactors[1])
        )
        
        if rebin:
            return (
                self.rebinner(smoothTemplates[0]),
                self.rebinner(smoothTemplates[1])
            )
        else:
            return smoothTemplates
    
    
    def _apply_deviation(self, deviation, scaleFactor=1.):
        """Apply given deviation to nominal template.
        
        Use the same deviation for all channels.
        
        Arguments:
            deviation:  Relative deviation to be applied.  Its shape is
                (nBinsAngle, nBinsMass).
            scaleFactor:  Optional scale factor to rescale deviation.
        
        Return value:
            New systematic template determined by the given deviation.
            Its shape is (nChannels, nBinsAngle, nBinsMass, 2).
        """
        
        totalScaling = 1 + scaleFactor * deviation
        totalScaling2 = totalScaling**2
        
        template = np.copy(self.nominal)
        
        # Apply the same deviation to each channel
        for channelTemplate in template:
            
            # Rescale central values
            channelTemplate[..., 0] *= totalScaling
            
            # Quadratic scaling for squared uncertainties
            channelTemplate[..., 1] *= totalScaling2
        
        
        return template
    
    
    def _compute_scale_factors(self):
        """Compute scale factors to match up and down variations.
        
        Determine the scale factors by minimizing the chi^2 difference
        between the systematic templates and ones obtained by applying
        the smoothed deviation, rescaled by that factors, to the nominal
        template.  The scale factor is the same for all bins.  Different
        channels are not summed up.  Computed with analysis binning.
        
        If the scale factors computed independently for up and down
        variations are compatible with each other (up to the sign)
        within their uncertainties (potentially scaled by a user-defined
        factor), the combined scale factor is used for both variations
        (up to the difference in signs).
        
        Arguments:
            None.
        
        Return value:
            None.
        """
        
        systSmoothRebinned = self.rebinner(
            self._apply_deviation(self.smoothAverageDeviation)
        )
        
        
        # Precompute pieces needed to compute scale factors for up and
        # down variations.  The scale factors are given by a / b, and
        # their uncertainties are 1 / sqrt(b).
        a, b = np.empty(2), np.empty(2)
        
        for iDirection in range(2):
            absDeviation = self.systsRebinned[iDirection][..., 0] - self.nominalRebinned[..., 0]
            absSmoothDeviation = systSmoothRebinned[..., 0] - self.nominalRebinned[..., 0]
            
            # Combined squared uncertainty
            unc2 = self.systsRebinned[iDirection][..., 1] + systSmoothRebinned[..., 1]
            
            # Computed using analytical results
            a[iDirection] = np.sum(absDeviation * absSmoothDeviation / unc2)
            b[iDirection] = np.sum(absSmoothDeviation**2 / unc2)
        
        
        # Store independent scale factors for bookkeeping
        self.rawScaleFactors = [(a[i] / b[i], 1 / math.sqrt(b[i])) for i in range(2)]
        
        
        # If the two scale factors are compatible (except for the sign),
        # use a combined one instead
        if (self.rawScaleFactors[0][0] + self.rawScaleFactors[1][0]) ** 2 < \
            self.sfCombThreshold ** 2 * (1 / b[0] + 1 / b[1]):
            
            sf = (a[0] - a[1]) / (b[0] + b[1])
            self.scaleFactors = (sf, -sf)
        
        else:
            self.scaleFactors = (self.rawScaleFactors[0][0], self.rawScaleFactors[1][0])
            
    
    
    def _smooth_impl_2d(self, bandwidth):
        """Peform 2D LOWESS smoothing.
        
        The bandwidth must be an array_like of length 2, which gives
        half-sizes of the windows, expressed in the number of bins, used
        along the dimensions of the angle and mass (in this order).
        """
        
        smoothAverageDeviation = lowess2d_grid(
            np.arange(0., self.nominal.shape[1], dtype=np.float64),
            np.arange(0., self.nominal.shape[2], dtype=np.float64),
            self.averageDeviation,
            bandwidth
        )
        
        return smoothAverageDeviation
    
    
    def _smooth_impl_mass(self, bandwidth):
        """Perform 1D LOWESS smoothing along the mass dimension.
        
        Treat different bins in the angle independently.  Given
        bandwidth is the half of the size of the window expressed in
        mass bins.
        """
        
        nBinsAngle = self.nominal.shape[1]
        
        
        # Smooth the deviation in each angular bin independently
        smoothAverageDeviation = np.empty_like(self.averageDeviation)
        
        for binAngle in range(nBinsAngle):
            deviationSlice = self.averageDeviation[binAngle]
            smoothAverageDeviation[binAngle] = lowess(
                range(len(deviationSlice)), deviationSlice, bandwidth
            )
        
        return smoothAverageDeviation
