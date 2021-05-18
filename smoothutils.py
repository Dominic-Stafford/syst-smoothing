"""Provides tools for smoothing of systematic variations."""

import itertools
import math
from operator import itemgetter
from uuid import uuid4

import numpy as np
import uproot

from lowess_numba import lowess


class Reader():
    """Reader for files for cross-validation."""
    
    def __init__(self, file_name):
        """Initialize from a ROOT file with templates.
        
        Arguments:
            file_name:  Path to a ROOT file with templates.
        """
        
        self.templates_file = uproot.open(file_name)
        
        # Extract the number of bins in histograms.  Since they all have
        # the same shape, pick one. Currently assuming only hists are saved in the file
        hist = self.templates_file.values()[0]
        if len(hist.to_numpy()) != 3:
            raise RuntimeError('Template of unexpected dimension {}.'.format(len(hist.to_numpy()) - 1))
        
        self.num_bins = len(hist.to_numpy()[1]) - 1
        self.num_partitions = len(hist.to_numpy()[2]) - 1
    
    
    def get_syst_names(self):
        """Provide set of systematic uncertainties included in the file.
        
        Assume that uncertainties are the same for all channels.
        """
        
        syst_names = set()
        
        for key in self.templates_file.Get(self.channels[0]).GetListOfKeys():
            name = key.GetName()
            
            if name.endswith('Up'):
                syst_names.add(name[:-2])
        
        return syst_names

    def _zero_counts(self):
        """Create empty (zero) template in NumPy representation."""
        
        return np.zeros((self.num_bins, 2)) 
    
    def read_counts_partitions(self, name, partitions):
        """Read partitions from set of templates.
        
        Read a set of templates with the given name and merge requested
        partitions.
        
        Arguments:
            name:  Name that identifies the set of histograms.
            partitions:  List of zero-based indices of partitions to be
                read.  The largest value must be smaller than
                self.num_partitions.
        
        Return value:
            Templates in the standard NumPy representation.
        """
        
        counts = self._zero_counts()

        hist = self.templates_file[name]
        
        if not hist:
            raise RuntimeError('Failed to read histogram "{}".'.format(name))
        
        counts[:, 0] = np.sum(hist.values()[:, partitions], axis=1)
        counts[:, 1] = np.sum(hist.variances()[:, partitions], axis=1)
        return counts
    
    
    def read_counts(self, name):
        """Read templates with the given name combining all partitions.
        
        Implement the method from the base class.
        """

        hist = self.templates_file[name]
        
        if not hist:
            raise RuntimeError('Failed to read histogram "{}".'.format(name))
        
        counts = np.empty((self.num_bins, 2))
        counts[:, 0] = np.sum(hist.values(), axis=1)
        counts[:, 1] = np.sum(hist.variances(), axis=1)
        
        return counts


class Rebinner1D:
    """An auxiliary class to rebin NumPy arrays along an axis.
    
    The array is supposed to represent a histogram of event counts with
    uncertainties.
    """
    
    def __init__(self, source_binning, target_binning, axis=None):
        """Initialize from source and target binnings.
        
        Arguments:
            source_binning, target_binning:  Source and target binnings.
                Must satisfy requirements listed for _build_rebin_map.
            axis:  Axis along which the rebinning is to be performed.
        """
        
        self.rebin_map = self._build_rebin_map(source_binning, target_binning)
        self.axis = axis
    
    
    @classmethod
    def from_map(cls, rebin_map, axis):
        """Construct from a rebin map."""
        
        self = cls.__new__(cls)
        self.rebin_map = rebin_map
        self.axis = axis
        
        return self
    
    
    def __call__(self, array, axis=None):
        """Create a new array with target binning along given axis."""
        
        if axis is None:
            axis = self.axis
        
        if axis is None:
            raise RuntimeError('No axis provided.')
        
        
        rebinned_shape = list(array.shape)
        rebinned_shape[axis] = len(self.rebin_map) - 1
        
        rebinned_array = np.zeros(rebinned_shape, dtype=array.dtype)
        
        for i in range(len(self.rebin_map) - 1):
            
            # An indexing object that selects a slice of rebinned_array
            # at index i along the given axes
            index = tuple([slice(None)] * axis + [i, ...])
            
            rebinned_array[index] = np.sum(
                np.take(array, range(self.rebin_map[i], self.rebin_map[i + 1]), axis=axis),
                axis=axis
            )
        
        return rebinned_array
    
    
    def translate_index(self, index_source):
        """Map bin in source binning into target binning.
        
        Arguments:
            index_source:  Index of the bin in the source binning.
                Negative values are allowed.
        
        Return value:
            Index of the bin in the target binning to which the given
            bin maps.
        """
        
        num_bins = self.rebin_map[-1]
        
        # Allow for negative indices as usual
        if index_source < 0:
            index_source += num_bins
        
        if index_source >= num_bins:
            raise IndexError
        
        return np.searchsorted(self.rebin_map, index_source, 'right') - 1
    
    
    @staticmethod
    def _build_rebin_map(source_binning, target_binning):
        """Construct mapping between source and target binning.
        
        Assume that all elements of the target binning are included into
        the source binning and both arrays are sorted.  Also assume that
        first and last bins from the source binning are included into
        the respective bins of the target binning.  Return a list such
        that element number i in it is the (zero-based) bin number of
        the first bin from the source binning included in bin i of the
        target binning.
        """
        
        source_binning = np.asarray(source_binning)
        target_binning = np.asarray(target_binning)
        
        
        # Use half of the width of the narrowest bin in the source
        # binning to match the binnings
        eps = np.min(source_binning[1:] - source_binning[:-1]) / 2
        
        start_indices_target = [0]
        
        for edge in target_binning[1:-1]:
            start_indices_target.append(np.searchsorted(source_binning, edge + eps) - 1)
        
        # The last element is the number of bins in the source binning
        # (which is one smaller than the number of edges)
        start_indices_target.append(len(source_binning) - 1)
        
        return start_indices_target




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
        
        rebinned_array = array
        
        for rebinner in self.rebinners:
            rebinned_array = rebinner(rebinned_array)
        
        return rebinned_array
    
    
    def translate_index(self, index_source):
        """Map bin in source binning into target binning.
        
        If for some axes no rebinners are available, bin indices along
        them are left unchanged.
        
        Arguments:
            index_source:  Multidimensional index of the bin in the
                source binning.  Negative values are allowed.
        
        Return value:
            Multidimensional index of the bin in the target binning to
            which the given bin maps.
        """
        
        index_target = list(index_source)
        
        for rebinner in self.rebinners:
            axis = rebinner.axis
            index_target[axis] = rebinner.translate_index(index_source[axis])
        
        return tuple(index_target)


class AdaptiveRebinner(RebinnerND):
    """Adaptive version of RebinnerND.
    
    Construct the rebin map using a reference template.  Its bins are
    merged recursively (while respecting the rectangular grid) until the
    maximal relative uncertainty computed for the sum of all channels
    becomes smaller than a given threshold.
    """
    
    def __init__(self, template, max_rel_unc=0.1):
        """Initialize from reference template.
        
        Arguments:
            template:  A template in the standard representation used
                by Reader classes, on which the binning is tuned.
            max_rel_unc:  Maximal relative uncertainty in a bin (after
                summing all channels) that determines the stopping
                condition for the optimization.
        """
        
        if template.ndim != 2:
            raise RuntimeError('Unexpected dimensionality {}.'.format(template.ndim))
        
        
        self.rebinners = [
            Rebinner1D.from_map(list(range(template.shape[0] + 1)), 0)
        ]
        
        template_rebinned = template
        
        while True:
            
            with np.errstate(invalid='ignore'):
                rel_unc_sq = template_rebinned[..., 1] / template_rebinned[..., 0] ** 2
            
            # Set the relative uncertainty to infinity in empty bins and
            # bins with very small content and zero error.  The latter
            # case occurs when a very small expected number of events is
            # set manually to previously empty bins, which is done for
            # some templates.
            rel_unc_sq[np.isnan(rel_unc_sq)] = np.inf
            rel_unc_sq[rel_unc_sq == 0.] = np.inf
            
            
            # Find bin with the largest uncertainty
            cur_max_unc_index = np.argmax(rel_unc_sq)
            
            if rel_unc_sq[cur_max_unc_index] < max_rel_unc ** 2:
                # The procedure has terminated
                break
            
            
            # Check the four possible regions to merge that contain the
            # bin with the largest uncertainty.  Choose from them the
            # one that contains the smallest effective number of events.
            # This way the merging that has the less drastic overall
            # effect is chosen.  The effective number of events equals
            # 1 divided by the squared relative uncertainty.  The
            # regions are identified by the relative position of one of
            # the four neighbouring bins.

            if cur_max_unc_index == 0:
                del self.rebinners[0].rebin_map[1]
            elif cur_max_unc_index == len(rel_unc_sq) - 1:
                del self.rebinners[0].rebin_map[cur_max_unc_index]
            elif rel_unc_sq[cur_max_unc_index + 1] > rel_unc_sq[cur_max_unc_index - 1]:
                del self.rebinners[0].rebin_map[cur_max_unc_index + 1]
            else:
                del self.rebinners[0].rebin_map[cur_max_unc_index]

            # Rebin the template
            template_rebinned = self(template)
            
            if len(template_rebinned) == 1:
                # Cannot rebin any further: there is only one bin left
                break



class RepeatedCV:
    """A class to provide inputs for repeated cross-validation.
    
    This class automatizes creation of test and training sets for
    cross-validation (CV).  The whole CV procedure can be repeated
    multiple times after reshuffling data.
    """
    
    def __init__(self, reader, k_cv):
        """Constructor from a reader object and number of CV partitions.
        
        Arguments:
            reader:  An instance of class inheriting from ReaderCV.
            k_cv:  Desired number of CV partitions.  Should normally be
                a divisor of the number of raw partitions in the reader.
        """
        
        self.reader = reader
        self.k_cv = k_cv
        
        # Booked total counts
        self.total_counts = {}
        
        # Order in which partitions will be iterated during
        # cross-validation
        self.partitions = list(range(reader.num_partitions))
        
        # Current partitions that constituent current test set and
        # corresponding counts for booked histograms
        self.test_partitions = None
        self.test_counts = {}
        
        # Complementary counts that define the training set
        self.train_counts = {}
    
    
    def book(self, names):
        """Specify names of histogram sets that will be used."""
        
        new_total_counts = {}
        
        for name in names:
            if name in self.total_counts:
                # Already booked.  No need to read again.
                new_total_counts[name] = self.total_counts[name]
            else:
                new_total_counts[name] = self.reader.read_counts(name)
        
        self.total_counts = new_total_counts
    
    
    def create_cv_partitions(self, k):
        """Create CV partitions.
        
        Define test and training partitions.
        
        Arguments:
            k:  Zero-based index of test CV partition.  Remaining CV
                partitions define the training data.
        """
        
        if k < 0 or k >= self.k_cv:
            raise IndexError('Illegal index.')
        
        cv_length = round(len(self.partitions) / self.k_cv)
        
        if k < self.k_cv - 1:
            self.test_partitions = self.partitions[k * cv_length : (k + 1) * cv_length]
        else:
            # Special treatment for the last CV partition in case the
            # number of CV folds is not aligned with the number of
            # partitions.  This way all data are utilized.
            self.test_partitions = self.partitions[k * cv_length :]
        
        
        # Update counts for booked histograms
        self.test_counts, self.train_counts = {}, {}
        
        for name, total_counts in self.total_counts.items():
            test_counts = self.reader.read_counts_partitions(name, self.test_partitions)
            self.test_counts[name] = test_counts
            self.train_counts[name] = total_counts - test_counts
    
    
    def get_counts_test(self, name):
        """Provide counts in the current test set."""
        
        if not self.test_counts:
            raise RuntimeError('CV partitions have not been created.')
        
        if name not in self.test_counts:
            raise RuntimeError('Counts with name "{}" have not been booked.'.format(name))
        
        return self.test_counts[name]
    
    
    def get_counts_train(self, name):
        """Provide counts in the current training set."""
        
        if not self.train_counts:
            raise RuntimeError('CV partitions have not been created.')
        
        if name not in self.train_counts:
            raise RuntimeError('Counts with name "{}" have not been booked.'.format(name))
        
        return self.train_counts[name]
    
    
    def shuffle(self):
        """Shuffle underlying raw partitions.
        
        This allows to repeat the full CV procedure anew.
        """
        
        np.random.shuffle(self.partitions)
        self.test_partitions = None
        self.test_counts = {}



class Smoother:
    """A class to smooth systematic variations.
    
    The relative deviation from the nominal template is averaged over
    all channels as well as up and down variations (taking into account
    the inversion in the shape) and then smoothed with a version of
    LOWESS algorithm.  After that, independent overall scale factors are
    applied to the smoothed relative deviation in each channel and for
    each direction.  If the scale factors for up and down variations are
    compatible within their uncertainties (apart from the sign), a
    combined scale factor is used instead.
    
    The procedure use templates with two binnings in the angle and mass,
    which are related by an instance of RebinnerND.  There are two modes
    of operation, controlled by a flag given at initialization.  The
    first mode is to use the fine binning for smoothing.  The overall
    scale factors are computed with the coarse binning (which in this
    case corresponds to the binning used in the statistical analysis).
    Final smoothed systematic variations are also rebinned for this
    coarse binning.  The second mode is to apply the rebinning before
    the smoothing.  This is useful when the number of available events
    is too small to construct the smoothed relative deviation reliably.
    As before, the overall scale factors are computed with the coarse
    binning.  The final smoothed systematic variations are upscaled to
    the fine binning.
    """
    
    def __init__(
        self, nominal, up, down, rebinner, rebin_for_smoothing=False, sf_comb_threshold=1.
    ):
        """Constructor from reference templates.
        
        Arguments:
            nominal, up, down:  Reference templates with fine binning.
                They must respect the format adopted in ReaderBase, i.e.
                their shapes must be (num_channels, num_bins_angle,
                num_bins_mass, 2).
            rebinner:  An instance of RebinnerND that implements the
                mapping to the coarse binning.
            rebin_for_smoothing:  Determines whether the smoothing
                should be performed with rebinned templates.
            sf_comb_threshold:  If the difference between scale factors
                for up and down (with flipped sign) variations computed
                independently is smaller than their combined uncertainty
                multiplied by the given factor, will use the same scale
                factor for both (up to the sign).
        """
        
        # Save templates in the binning intended for smoothing.  Also
        # store the original nominal template.
        if rebin_for_smoothing:
            self.nominal_input = nominal
            self.nominal = rebinner(nominal)
            self.up = rebinner(up)
            self.down = rebinner(down)
        else:
            self.nominal_input = nominal
            self.nominal = nominal
            self.up = up
            self.down = down
        
        self.rebinner = rebinner
        self.rebin_for_smoothing = rebin_for_smoothing
        
        
        # Sanity check: the procedure does not work when the nominal
        # template is empty in some bins for all channels
        if np.any(self.nominal[..., 0] == 0.):
            raise RuntimeError('Smoothing is not supported for nominal templates with empty bins.')
        
        
        # Precompute relative deviation from the nominal template,
        # averaged over channels and up and down variations.  The sign
        # is chosen in such a way that obtained deviation corresponds to
        # the up variation.  The shape of produced array is
        # (num_bins_angle, num_bins_mass).
        self.average_deviation = 0.5 * (self.up[:, 0] - self.down[:, 0]) / self.nominal[:, 0]
        
        
        # Compute uncertainties on the relative deviation using naive
        # error propagation formulas.  Uncertainties of the three input
        # templates are assumed to be independent.  Note that for
        # smoothing of the average deviation only the relative
        # uncertainty in different bins is relavant.
        self.average_deviation_unc2 = (self.up[:, 1] + self.down[:, 1] + ((self.up[:, 0] - self.down[:, 0]) / self.nominal[:, 0]) ** 2
                                       * self.nominal[:, 1]) / (2 * self.nominal[:, 0]) ** 2
        
        
        # Precompute templates with the binning used for the computation
        # of the overall scale factors
        if rebin_for_smoothing:
            self.nominal_bin_sf = self.nominal
            self.systs_bin_sf = (self.up, self.down)
        else:
            self.nominal_bin_sf = rebinner(self.nominal)
            self.systs_bin_sf = (rebinner(self.up), rebinner(self.down))
        
        self.smooth_average_deviation = None
        
        
        self.sf_comb_threshold = sf_comb_threshold
        self.scale_factors, self.raw_scale_factors = None, None
            

    def smooth(self, bandwidth):
        """Perform LOWESS smoothing using given bandwidth.
        
        Arguments:
            bandwidth:  Bandwidth used for smoothing.  Exact meaining
                depends on the algorithm, see documentation for
                respective methods.
            algo:  Smoothing algorithm.  Supported are '2D' and 'mass'.
        
        Return value:
            A pair of templates that represent up and down smoothed
            variations.  Their shapes are (num_channels, num_bins_angle,
            num_bins_mass, 2).  The uncertainty is computed from the
            uncertainty of the nominal template.
        
        Add to 'self' smoothed averaged relative deviation and scale
        factors for up and down deviations.
        """
        
        # Construct smoothed average deviation
        self.smooth_average_deviation = self._smooth_1d(bandwidth)
        
        
        # Compute scale factors for up and down variations
        self._compute_scale_factors()
        
        
        # Construct smoothed systematic templates
        if self.rebin_for_smoothing:
            # Apply smooth deviation obtained with the coarse binning to
            # the input nominal template, which has a finer binning
            systs_smooth = (
                self._apply_deviation(
                    self.nominal_input, self.smooth_average_deviation,
                    scale_factor=self.scale_factors[0]
                ),
                self._apply_deviation(
                    self.nominal_input, self.smooth_average_deviation,
                    scale_factor=self.scale_factors[1]
                )
            )
            
            return systs_smooth
        
        else:
            # Apply smooth deviation obtained with the dense binning to
            # the input nominal template, which has the same binning.
            # Then rebin thus constructed smooth systematic variations.
            systs_smooth = (
                self._apply_deviation(
                    self.nominal, self.smooth_average_deviation, scale_factor=self.scale_factors[0]
                ),
                self._apply_deviation(
                    self.nominal, self.smooth_average_deviation, scale_factor=self.scale_factors[1]
                )
            )
            
            return (
                self.rebinner(systs_smooth[0]),
                self.rebinner(systs_smooth[1])
            )
    
    
    def _compute_scale_factors(self):
        """Compute scale factors to match up and down variations.
        
        Determine the scale factors by minimizing the chi^2 difference
        between the systematic templates and ones obtained by applying
        the smoothed deviation, rescaled by that factors, to the nominal
        template.  The scale factor is the same for all bins.  Different
        channels are not summed up.  Computed with the binning used for
        smoothing.
        
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
        
        # Construct template for smooth up variation with the overall
        # scale factor of 1.  This is an auxiliary template needed for
        # the computation of the actual scale factors.
        up_smooth = self._apply_deviation(self.nominal, self.smooth_average_deviation)
        
        if not self.rebin_for_smoothing:
            # Smoothing has been performed with the dense binning.  Then
            # rebin the template now.
            up_smooth = self.rebinner(up_smooth)
        
        
        # Precompute pieces needed to compute scale factors for up and
        # down variations.  The scale factors are given by a / b, and
        # their uncertainties are 1 / sqrt(b).
        a, b = np.empty(2), np.empty(2)
        
        for direction in range(2):
            abs_deviation = self.systs_bin_sf[direction][..., 0] - self.nominal_bin_sf[..., 0]
            abs_smooth_deviation = up_smooth[..., 0] - self.nominal_bin_sf[..., 0]
            
            # Combined squared uncertainty
            unc2 = self.systs_bin_sf[direction][..., 1] + up_smooth[..., 1]
            
            # Computed using analytical results
            a[direction] = np.sum(abs_deviation * abs_smooth_deviation / unc2)
            b[direction] = np.sum(abs_smooth_deviation ** 2 / unc2)
        
        
        # Store independent scale factors for bookkeeping
        self.raw_scale_factors = [(a[i] / b[i], 1 / math.sqrt(b[i])) for i in range(2)]
        
        
        # If the two scale factors are compatible (except for the sign),
        # use a combined one instead
        if (self.raw_scale_factors[0][0] + self.raw_scale_factors[1][0]) ** 2 < \
            self.sf_comb_threshold ** 2 * (1 / b[0] + 1 / b[1]):
            
            sf = (a[0] - a[1]) / (b[0] + b[1])
            self.scale_factors = (sf, -sf)
        
        else:
            self.scale_factors = (self.raw_scale_factors[0][0], self.raw_scale_factors[1][0])
    
    
    def _apply_deviation(self, nominal, deviation, scale_factor=1.):
        """Apply relative deviation to given nominal template.
        
        Use the same deviation for all channels.  The deviation can be
        given with a coarser binning in angle and mass than used in the
        nominal template.  In that case assume that the two binnings are
        related by self.rebinner.
        
        Arguments:
            nominal:  Nominal template.  Its shape is (num_bins_angle,
                num_bins_mass).
            deviation:  Relative deviation to be applied.  Its shape is
                (num_bins_angle, num_bins_mass), but there might be
                fewer bins along each axis than in the nominal template.
            scale_factor:  Optional scale factor to rescale deviation.
        
        Return value:
            New systematic template determined by the given deviation.
            Its shape is (num_channels, num_bins_angle, num_bins_mass,
            2), where numbers of bins along angle and mass axes are
            given by the nominal template.
        """
        
        if len(nominal) != len(deviation):
            # Numbers of bins along angle and mass axes in the given
            # nominal template do not match ones in the deviation.  The
            # deviation mush have been computed with a coarser binning.
            # Upscale it by duplicating bins.
            deviation_coarse = deviation
            deviation = np.empty(len(nominal))
            
            for bin_i in range(len(nominal)):
                deviation[bin_i] = deviation_coarse[self.rebinner.translate_index([bin_i])]
                # The rebinner uses the standard NumPy representation,
                # which contains also an index for the channel.  Add a
                # dummy one here.
            
        
        total_scaling = 1 + scale_factor * deviation
        total_scaling_sq = total_scaling ** 2
        
        template = np.copy(nominal)
        
        template[..., 0] *= total_scaling
        template[..., 1] *= total_scaling_sq
        
        return template


    def _smooth_1d(self, bandwidth):
        smooth_average_deviation = lowess(
            np.arange(len(self.average_deviation), dtype=np.float64), self.average_deviation, bandwidth,
            weights = 1 / self.average_deviation_unc2)
        
        return smooth_average_deviation



if __name__ == '__main__':
    
    # Tests for AdaptiveRebinner
    def print_rel_unc(template):
        template_sum_channels = np.sum(template, axis=0)
        with np.errstate(invalid='ignore'):
            rel_unc = np.sqrt(template_sum_channels[..., 1]) / template_sum_channels[..., 0]
        print(rel_unc)
    
    def print_rebin_map(rebinner):
        for r in rebinner.rebinners:
            print(r.rebin_map, end=' ')
        print()
    
    template = np.empty((2, 2, 5, 2))
    template[..., 0] = 1.
    
    for bin_channel, bin_angle in itertools.product(
        range(template.shape[0]), range(template.shape[1])
    ):
        template[bin_channel, bin_angle, :, 1] = np.asarray([0.01, 0.05, 0.1, 0.2, 0.5]) ** 2
    
    rebinner = AdaptiveRebinner(template)
    print_rel_unc(template)
    print_rebin_map(rebinner)
    print_rel_unc(rebinner(template))
    print()
    
    
    template[0, 1, 4, :] = [0., 0.]
    template[1, 1, 4, :] = [0., 0.]
    
    rebinner = AdaptiveRebinner(template)
    print_rel_unc(template)
    print_rebin_map(rebinner)
    print_rel_unc(rebinner(template))
    print()
