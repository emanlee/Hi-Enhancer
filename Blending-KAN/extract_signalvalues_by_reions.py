#  
"""
Extract ChIP-seq/DNase-seq signal features from BigWig files.

regions file example (seperated by Tab):
chr1	1005800	1007301
chr1	1005960	1007461
chr1	1304100	1305601
chr1	1304304	1305805


$python extract_signalvalues_by_reions.py \
    -b path/to/bigWig \
    -r path/to/regions.bed \
    -o path/to/output_features.csv \
    -w 4000 \
    -n 400

Example£¨Take DNase-seq for extraction of HCT116 as an example£©:
$python extract_signalvalues_by_reions.py     -b HCT116.DNase-seq.bigWig     -r HCT116.positive.bed     -o output_features.csv     -w 4000     -n 400


"""
import os
import argparse
import numpy as np
import pyBigWig
import pybedtools
from pybedtools import BedTool
import csv

# ----- Parse command line arguments ----- #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract signal values/features from BigWig files.")
    parser.add_argument("-b", "--bigwig", required=True, help="Input file: Path to the ChIP-seq/DNase-seq BigWig file.")
    parser.add_argument("-r", "--regions", required=True, help="Input file: Path to the BED file containing genomic regions.")
    parser.add_argument("-o", "--output", required=True, help="Output file: Output file path for the feature matrix (CSV format).")
    parser.add_argument("-w", "--window_size", type=int, default=4000, help="Total window size for each region (default: 4000).")
    parser.add_argument("-n", "--num_bins", type=int, default=400, help="Number of bins to divide the window into (default: 400).")
    return parser.parse_args()

# ----- Extract signal features from BigWig file ----- #
def extract_signal_features(bigwig_path, regions_bed, window_size, num_bins):
    # Load BigWig file
    bw = pyBigWig.open(bigwig_path)
    if bw is None:
        raise ValueError(f"Failed to open BigWig file: {bigwig_path}")

    # Load regions from BED file
    regions = BedTool(regions_bed)

    # Initialize feature matrix
    feature_matrix = []

    # Iterate over each region
    for region in regions:
        chrom, start, end = region.chrom, region.start, region.end
        region_size = end - start

        # Adjust region size to match window_size
        if region_size < window_size:
            pad = (window_size - region_size) // 2
            start = max(0, start - pad)
            end = start + window_size

        # Extract signal values
        try:
            signal_values = bw.values(chrom, start, end)
        except Exception as e:
            print(f"Error extracting signal for {chrom}:{start}-{end}: {e}")
            signal_values = [0.0] * window_size

        # Handle missing values
        signal_values = np.nan_to_num(signal_values, nan=0.0)

        # Bin the signal values
        if len(signal_values) != window_size:
            signal_values = np.interp(
                np.linspace(0, len(signal_values) - 1, window_size),
                np.arange(len(signal_values)),
                signal_values
            )

        binned_signal = np.array_split(signal_values, num_bins)
        binned_signal = [np.mean(bin) for bin in binned_signal]

        feature_matrix.append(binned_signal)

    return np.array(feature_matrix)

# ----- Save feature matrix to CSV file with column headers ----- #
def save_feature_matrix(feature_matrix, output_path, num_bins):
    # Generate column headers
    column_headers = [f"x{i}" for i in range(num_bins)]

    # Save to CSV file
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write column headers
        writer.writerow(column_headers)
        # Write feature matrix
        writer.writerows(feature_matrix)

    print(f"Feature matrix saved to {output_path}")

# ----- Main function ----- #
def main():
    args = parse_arguments()

    print("Extracting signal features...")
    feature_matrix = extract_signal_features(
        bigwig_path=args.bigwig,
        regions_bed=args.regions,
        window_size=args.window_size,
        num_bins=args.num_bins
    )

    print("Saving feature matrix...")
    save_feature_matrix(feature_matrix, args.output, args.num_bins)

if __name__ == "__main__":
    main()