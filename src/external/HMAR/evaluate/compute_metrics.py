# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

from utils.evaluation import compute_metrics
import prettytable as pt
import argparse
import dist
import sys
import os
from utils import misc


def main(npz_file, experiment):
    results = pt.PrettyTable()
    results.field_names = ["Experiment", "FID", "sFID", "IS", "Precision", "Recall"]

    FID, sFID, IS, prec, recall = compute_metrics(sample_npz=npz_file)
    results.add_row([experiment, FID, sFID, IS, prec, recall])

    print(results)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Compute quantitative metrics for samples generated from a checkpoint"
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            help="checkpoint for which to compute metrics",
        )
        args = parser.parse_args()

        npz_file = os.path.join(os.getcwd(), f"samples-{args.checkpoint}.npz")
        main(npz_file, args.checkpoint)

        os.remove(npz_file)

    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
