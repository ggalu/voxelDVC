# -*- coding: utf-8 -*-
"""run_ground_truth_pipeline.py: drive the 4-stage known-ground-truth DVC
accuracy check end to end, using the exact same preprocess/run_dvc pipeline
used on real (unknown-ground-truth) data. Each stage is run as a subprocess
via `-m`, so it works the same whether the package is installed or run from
a checkout:

  1. voxeldvc.ground_truth.prepare_ground_truth -- generate the reference/deformed image pair
  2. voxeldvc.preprocess                        -- affine pre-alignment + overlap crop
  3. voxeldvc.run_dvc                           -- multiscale DVC correlation
  4. voxeldvc.ground_truth.analyse_ground_truth -- compare recovered vs. ground-truth displacement

Usage
-----
  voxeldvc ground-truth --output-dir DIR [--gt-ref-file PATH] \
      [--gt-displacement-file PATH | --tensile-strain STRAIN] \
      [--h H] [--l0-factor L0_FACTOR]

  --gt-ref-file PATH        forwarded to voxeldvc.ground_truth.prepare_ground_truth:
                            reference image .npy file, shape (N,N,N) (default:
                            assets/GT_z+0.01/ref.npy).
  --gt-displacement-file PATH
                            forwarded to voxeldvc.ground_truth.prepare_ground_truth:
                            ground-truth displacement .npy file, shape
                            (N+1,N+1,N+1,3) (default:
                            assets/GT_z+0.01/U.npy). See
                            that module's docstring for the required
                            image/displacement dimension relationship.
                            Mutually exclusive with --tensile-strain.
  --tensile-strain STRAIN  forwarded to voxeldvc.ground_truth.prepare_ground_truth:
                            impose a synthetic uniaxial strain instead of
                            --gt-displacement-file. Mutually exclusive with
                            --gt-displacement-file.
  --h H                    DVC element edge length in voxels (default: 8),
                            forwarded to voxeldvc.preprocess.
  --l0-factor L0_FACTOR    regularization length l0 = L0_FACTOR * h, in voxels
                            (default: 4.5), forwarded to voxeldvc.run_dvc.
  --output-dir DIR         root output directory (required, no default).
                            Stage 1 writes to <dir>/gt_input; stages 2-4
                            write to <dir>.
"""

import argparse
import os
import subprocess
import sys

from .dvc_defaults import DEFAULT_H, DEFAULT_L0_FACTOR


def run_stage(module, args):
    full_args = [sys.executable, '-m', module] + args
    print("\n" + "=" * 70)
    print("STAGE: " + " ".join(full_args))
    print("=" * 70)
    subprocess.run(full_args, check=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gt-ref-file', type=str, default=None, dest='gt_ref_file',
                   help="Forwarded to ground_truth/prepare_ground_truth.py "
                        "(default: its own assets/GT_z+0.01 default).")
    group = p.add_mutually_exclusive_group()
    group.add_argument('--gt-displacement-file', type=str, default=None, dest='gt_displacement_file',
                        help="Forwarded to ground_truth/prepare_ground_truth.py "
                             "(default: its own assets/GT_z+0.01 default). "
                             "Mutually exclusive with --tensile-strain.")
    group.add_argument('--tensile-strain', type=float, default=None, dest='tensile_strain',
                        help="Forwarded to ground_truth/prepare_ground_truth.py. "
                             "Mutually exclusive with --gt-displacement-file.")
    p.add_argument('--h', type=int, default=DEFAULT_H, dest='h',
                   help=f"DVC element edge length in voxels (default: {DEFAULT_H}), "
                        "forwarded to preprocess.py.")
    p.add_argument('--l0-factor', type=float, default=DEFAULT_L0_FACTOR, dest='l0_factor',
                   help="Regularization length l0 = l0_factor * h, in voxels "
                        f"(default: {DEFAULT_L0_FACTOR}), forwarded to run_dvc.py.")
    p.add_argument('--output-dir', type=str, required=True,
                   dest='output_dir',
                   help="Root output directory (required; relative to the "
                        "current working directory). "
                        "Stage 1 writes to <dir>/gt_input; stages 2-4 write to <dir>.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    output_dir_abs = os.path.abspath(args.output_dir)
    gt_dir_abs = os.path.join(output_dir_abs, 'gt_input')

    # --- stage 1: voxeldvc.ground_truth.prepare_ground_truth ---
    stage1 = ['--output-dir', gt_dir_abs]
    if args.gt_ref_file is not None:
        stage1 += ['--gt-ref-file', args.gt_ref_file]
    if args.gt_displacement_file is not None:
        stage1 += ['--gt-displacement-file', args.gt_displacement_file]
    if args.tensile_strain is not None:
        stage1 += ['--tensile-strain', str(args.tensile_strain)]
    run_stage('voxeldvc.ground_truth.prepare_ground_truth', stage1)

    # --- stage 2: voxeldvc.preprocess ---
    run_stage('voxeldvc.preprocess', [
        os.path.join(gt_dir_abs, 'ref.npy'),
        os.path.join(gt_dir_abs, 'deformed.npy'),
        '--h', str(args.h),
        '--output-dir', output_dir_abs,
    ])

    # --- stage 3: voxeldvc.run_dvc ---
    run_stage('voxeldvc.run_dvc', [output_dir_abs, '--l0-factor', str(args.l0_factor)])

    # --- stage 4: voxeldvc.ground_truth.analyse_ground_truth ---
    run_stage('voxeldvc.ground_truth.analyse_ground_truth', [
        '--work-dir', output_dir_abs,
        '--gt-dir', gt_dir_abs,
    ])

    print("\n" + "=" * 70)
    print("✓ GROUND-TRUTH PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  gt_input : {gt_dir_abs}")
    print(f"  work_dir : {output_dir_abs}")
    print(f"  logs     : {os.path.join(output_dir_abs, 'run_log.txt')}, "
          f"{os.path.join(output_dir_abs, 'ground_truth_comparison_log.txt')}")


if __name__ == "__main__":
    main()
