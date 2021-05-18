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
mpl.use("agg")
from matplotlib import pyplot as plt


def parse_file(path):
    """Parse file with output of a single job."""
    
    with open(path) as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        raise RuntimeError(f"File {path} is empty.")
    
    
    # Drop CSV header
    del lines[0]
    
    
    split_lines = [line.split(",") for line in lines]
    
    for i in range(len(split_lines)):
        if len(split_lines[i]) != 6:
            raise RuntimeError("In file {path}, failed to parse line {lines[i]}.")
    
    
    # Template and variation are the same for one file.  Extract them
    # from the first line.
    template = split_lines[0][0]
    variation = split_lines[0][1]
    
    errors = np.asarray([[float(l[2]), float(l[4])] for l in split_lines])
    
    return template, variation, errors


if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(epilog=__doc__)
    arg_parser.add_argument("inputs", help="Directory with outputs of individual jobs")
    arg_parser.add_argument(
        "-o", "--output", default="bandwidths.csv",
        help="Name for output file with chosen bandwidths"
    )
    arg_parser.add_argument("-p", "--plot", action="store_true", help="Make plots")
    arg_parser.add_argument(
        "--fig-dir", default="fig/CV",
        help="Directory for produced figures"
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
      
        
        template, variation, errors = parse_file(path)
        bandwidth = errors[np.argmin(errors[:, 1])][0]
        
        optimal_badwidths.append((template, variation, bandwidth))
        plt.plot(errors[:, 0], errors[:, 1])
        plt.savefig(os.path.join(args.fig_dir, template + "_" + variation + ".pdf"))
        plt.close()
        
    
    optimal_badwidths.sort()
    
    
    with open(args.output, "w") as out_file:
        out_file.write("#Template,Variation,h_\n")
      
        for template, variation, bandwidth in optimal_badwidths:
            out_file.write("{},{},{:g}\n".format(template, variation, bandwidth))