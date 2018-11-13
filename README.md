# Smoothing of systematic variations

Code to suppress statistical fluctuations in systematic variations described by Monte-Carlo templates. The fluctuations appear when templates defining up and down variations are produced from independent MC samples, but also in case of uncertainties that allow events to migrate between different bins of the distribution, such as uncertainties in jet corrections.

The fluctuations are suppressed by smoothing relative variations with respect to  the nominal distribution. The smoothing is performed with the help of the [LOWESS](https://en.wikipedia.org/wiki/Local_regression) algorithm.

Code here is used for smoothing of JME variations in signal templates in HIG-17-027.

Computing environment [LCG_94python3](http://lcginfo.cern.ch/release/94python3/) is used. Cython module is compiled with

```sh
python setup.py build_ext --inplace
```