# Smoothing of systematic variations

This repository contains code to suppress statistical fluctuations in systematic variations described by Monte-Carlo templates. The fluctuations appear when templates defining up and down variations are produced from independent MC samples, but also in case of uncertainties that allow events to migrate between different bins of the distribution, such as uncertainties in jet corrections. As discussed in [this talk](https://indico.in2p3.fr/event/19290/contributions/75880/), the fluctuations can lead to severe unphysical constraints on the corresponding nuisance parameters of the measurement model.

Scripts included here were used in [CMS HIG-17-027](http://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-17-027/index.html) to suppress fluctuations in variations related to calibration of jets and missing p<sub>T</sub> in signal templates. A somewhat similar although not identical approach was used for fluctuations in the combined background. The procedure is documented in analysis note [AN-18-077](http://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2018/077) (restricted to CMS members).


## Overview of the algorithm

The algorithm operates with the relative deviations with respect to the nominal distribution. Bins in the tails of the distribution can contain insufficient number of events. To deal with this, underpopulated bins are merged recursively (while respecting the rectangular grid) until their relative uncertainties become smaller than a given threshold. The smoothing procedure will assign the same relative deviation to all bins that have been merged together.

The relative deviation is smoothed using a version of the [LOWESS](https://en.wikipedia.org/wiki/Local_regression) algorithm. As the independent coordinate it uses bin indices instead of the position of, for instance, the bin centre in the physical coordinate system. Deviations for ‘up’ and ‘down’ variations are forced to be symmetric in shape, although they are allowed to differ in the overall size. In addition, relative deviations in &mu;&thinsp;+&thinsp;jets and e&thinsp;+&thinsp;jets channels are taken to be the same. The smoothed relative deviation is constructed while respecting these conditions. For each direction of the variation it is then rescaled independently to match the corresponding input relative deviation best. If the resulting scale factors are compatible within their uncertainties (except for the sign), the final smoothed variation is forced to be fully symmetric.

The free parameter of the LOWESS algorithm is the bandwidth along each axis. It is tuned using repeated cross-validation.


## Software

Computing environment used is [LCG_94python3](http://lcginfo.cern.ch/release/94python3/). For performance reasons, LOWESS is implemented with a Cython model, which needs to be compiled with the following command:

```sh
python setup.py build_ext --inplace
```

First, the optimal bandwidths for all signal templates and all systematic variations need to be determined. This is done with a command like

```sh
cross_validation_sgn.py sgnHistsCV.root ggA_pos-sgn-5pc-M500 CMS_scale_j_13TeV_FlavorQCD -r 2000
```

It repeats 10-fold cross-validation for the given signal template and systematic variation 2k times. File `sgnHistsCV.root` contains signal templates split into partitions for cross-validation. Resulting mean errors for each probed bandwidth are stored in a text file. This should normally be executed on a batch system for various templates and variations in parallel. When the computation is done for all combinations, the results are analysed by script `analyse_cv_sgn.py`, which can also plot mean errors for different bandwidths.

When optimal bandwidths are known, the smoothing can be applied with

```sh
smooth_templates_sgn.py templates_orig.root -b bandwidths.csv -o templates_smooth.root
```

Here `bandwidths.csv` is the file created by script `analyse_cv_sgn.py` and `templates_orig.root` is the input file with all unrolled templates. The script reads the input file and applies the smoothing to systematic variations in signal templates as appropriate. Signal templates with variations that do not need to be smoothed as well as background templates are copied to the output file unchanged.

Finally, the difference between input and smoothed variations can be illustrated with plots produced with commands like

```sh
plot_smooth_templates_sgn.py templates_orig.root templates_smooth.root -s ggA_pos-sgn-5pc-M500
```
