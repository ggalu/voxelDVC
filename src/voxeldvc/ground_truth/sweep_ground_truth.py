# -*- coding: utf-8 -*-
"""sweep_ground_truth.py: run run_ground_truth_pipeline.py once per (h,
l0_factor) combination and tabulate node RMS displacement error across the
sweep -- the scripted version of the h/l0 sensitivity sweep described in
docs/TUTORIAL.md §9 and CLAUDE.md.

For each combination, the full 4-stage pipeline (prepare-ground-truth ->
preprocess -> run -> analyse-ground-truth) is run unmodified via
run_ground_truth_pipeline.main(), writing into its own subdirectory
`<output-root>/h<h>_l0f<l0_factor>/`. Two RMS metrics are read back from that
subdirectory's U_error.npy + node masks (the same arrays
ground_truth/analyse_ground_truth.py itself saves):

  - recoverable-node RMS (over ~truncated: GT-deformed support in g), and
  - valid-node RMS (recoverable minus the regularizer-biased boundary shell
    and overlap-lost nodes, from node_valid_mask.npy) -- the fair measure of
    DVC accuracy; recoverable still over-counts the boundary shell.

The final `dU/U` convergence value is scraped from run_log.txt.

Usage
-----
  voxeldvc sweep-ground-truth --gt-ref-file PATH \
      (--gt-displacement-file PATH | --tensile-strain STRAIN) \
      --output-root DIR [--h H_LIST] [--l0-factor L0_FACTOR_LIST] [--csv PATH]

  --gt-ref-file PATH        required. Reference image .npy file, shape
                            (N,N,N), forwarded to every combo's
                            prepare-ground-truth stage.
  --gt-displacement-file PATH
                            ground-truth displacement .npy file, shape
                            (N+1,N+1,N+1,3) -- one more grid point per axis
                            than --gt-ref-file; see
                            ground_truth/prepare_ground_truth.py's docstring.
                            Exactly one of this or --tensile-strain is
                            required.
  --tensile-strain STRAIN   forwarded to every combo's
                            ground_truth/prepare_ground_truth.py stage,
                            instead of --gt-displacement-file. Exactly one
                            of this or --gt-displacement-file is required.
  --h H_LIST               comma-separated element edge lengths h to sweep,
                            in voxels (default: 4,8,16).
  --l0-factor L0_FACTOR_LIST
                            comma-separated l0_factor values to sweep
                            (l0 = l0_factor * h); default: 2.25,4.5,9.
  --output-root DIR         required. Root output directory; each combo
                            writes to <root>/h<h>_l0f<l0_factor>/.
  --csv PATH                optional path to also write the summary table as CSV.

A combo whose recoverable-node RMS exceeds DIVERGED_RMS_THRESHOLD voxels is
flagged DIVERGED in the printed table (the GN solve completes its iteration
budget without crashing even when badly under-regularized, so divergence
shows up as an implausibly large RMS rather than a raised exception).
"""

import argparse
import csv as csv_module
import os
import re
import subprocess
import sys

import numpy as np

from .. import run_ground_truth_pipeline
from ..dvc_defaults import DEFAULT_H_SWEEP, DEFAULT_L0_FACTOR_SWEEP

DIVERGED_RMS_THRESHOLD = 1.0  # voxels; see module docstring

_DU_OVER_U_RE = re.compile(r'dU/U=([0-9.eE+-]+)')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gt-ref-file', type=str, required=True, dest='gt_ref_file',
                   help="Required. Reference image .npy file, shape (N,N,N), forwarded to "
                        "every combo's prepare-ground-truth stage.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--gt-displacement-file', type=str, default=None, dest='gt_displacement_file',
                        help="Ground-truth displacement .npy file, shape (N+1,N+1,N+1,3), "
                             "forwarded to every combo's prepare-ground-truth stage. Exactly "
                             "one of this or --tensile-strain is required.")
    group.add_argument('--tensile-strain', type=float, default=None, dest='tensile_strain',
                        help="Forwarded to every combo's ground_truth/prepare_ground_truth.py "
                             "stage, instead of --gt-displacement-file. Exactly one of this "
                             "or --gt-displacement-file is required.")
    p.add_argument('--h', type=str, default=DEFAULT_H_SWEEP, dest='h_values',
                   help=f"Comma-separated element edge lengths h to sweep, in voxels "
                        f"(default: {DEFAULT_H_SWEEP}).")
    p.add_argument('--l0-factor', type=str, default=DEFAULT_L0_FACTOR_SWEEP, dest='l0_factors',
                   help="Comma-separated l0_factor values to sweep (l0 = l0_factor * h) "
                        f"(default: {DEFAULT_L0_FACTOR_SWEEP}).")
    p.add_argument('--output-root', type=str, required=True, dest='output_root',
                   help="Required. Root output directory; each combo writes to "
                        "<root>/h<h>_l0f<l0_factor>/.")
    p.add_argument('--csv', type=str, default=None, dest='csv_path',
                   help="Optional path to also write the summary table as CSV.")
    return p.parse_args(argv)


def parse_final_du_over_u(run_log_path):
    """Best-effort scrape of the last `dU/U=...` value in run_log.txt -- the
    GN solve's convergence measure at its very last iteration (finest
    multiscale level).
    """
    try:
        with open(run_log_path) as fh:
            text = fh.read()
    except OSError:
        return float('nan')
    matches = _DU_OVER_U_RE.findall(text)
    return float(matches[-1]) if matches else float('nan')


def _rms_max(error, mask):
    """(rms over all 3 components, max nodal |err|) over the selected nodes,
    or (nan, nan) if the selection is empty."""
    err_m = error[mask]
    if err_m.size == 0:
        return float('nan'), float('nan')
    return (float(np.sqrt(np.mean(err_m ** 2))),
            float(np.max(np.linalg.norm(err_m, axis=-1))))


def summarize_combo(work_dir):
    """Recompute the RMS/max-error stats ground_truth/analyse_ground_truth.py
    prints, from the arrays it saves -- both the truncation-recoverable set
    and the mask-valid set (recoverable minus the regularizer-biased boundary
    shell and overlap-lost nodes; see node_valid_mask.npy). The valid metric
    is the fair measure of DVC accuracy; recoverable still over-counts the
    boundary shell. node_valid_mask.npy is absent for a work_dir produced
    before that mask existed, in which case the valid stats are nan.
    """
    error = np.load(os.path.join(work_dir, 'U_error.npy'))
    truncated = np.load(os.path.join(work_dir, 'node_truncated_mask.npy'))
    recoverable = ~truncated
    rms, max_err = _rms_max(error, recoverable)

    valid_path = os.path.join(work_dir, 'node_valid_mask.npy')
    if os.path.exists(valid_path):
        valid = np.load(valid_path)
        n_valid = int(valid.sum())
        rms_valid, max_err_valid = _rms_max(error, valid)
    else:
        n_valid, rms_valid, max_err_valid = 0, float('nan'), float('nan')

    return dict(
        n_recoverable=int(recoverable.sum()),
        n_total=int(truncated.size),
        rms=rms,
        max_err=max_err,
        n_valid=n_valid,
        rms_valid=rms_valid,
        max_err_valid=max_err_valid,
        du_final=parse_final_du_over_u(os.path.join(work_dir, 'run_log.txt')),
        status='ok',
    )


def run_sweep(h_values, l0_factors, gt_ref_file, gt_displacement_file, tensile_strain, output_root):
    results = []
    for h in h_values:
        for l0_factor in l0_factors:
            combo_dir = os.path.join(output_root, f"h{h}_l0f{l0_factor:g}")
            pipeline_args = ['--h', str(h), '--l0-factor', str(l0_factor),
                              '--output-dir', combo_dir]
            if gt_ref_file is not None:
                pipeline_args += ['--gt-ref-file', gt_ref_file]
            if gt_displacement_file is not None:
                pipeline_args += ['--gt-displacement-file', gt_displacement_file]
            if tensile_strain is not None:
                pipeline_args += ['--tensile-strain', str(tensile_strain)]

            print("\n" + "#" * 70)
            print(f"# SWEEP COMBO: h={h}  l0_factor={l0_factor:g}  (l0={l0_factor * h:g} vox)")
            print("#" * 70)
            try:
                run_ground_truth_pipeline.main(pipeline_args)
                stats = summarize_combo(combo_dir)
            except subprocess.CalledProcessError as exc:
                print(f"!! combo h={h}, l0_factor={l0_factor:g} FAILED: {exc}", file=sys.stderr)
                stats = dict(n_recoverable=0, n_total=0, rms=float('nan'), max_err=float('nan'),
                             n_valid=0, rms_valid=float('nan'), max_err_valid=float('nan'),
                             du_final=float('nan'), status='FAILED')

            results.append(dict(h=h, l0_factor=l0_factor, l0=l0_factor * h, **stats))
    return results


def _best(results, key):
    """Combo with the lowest finite `key` among non-diverged, ok combos."""
    best = None
    for r in results:
        if r['status'] != 'ok' or r['rms'] > DIVERGED_RMS_THRESHOLD:
            continue
        if np.isnan(r[key]):
            continue
        if best is None or r[key] < best[key]:
            best = r
    return best


def print_summary_table(results):
    print("\n" + "=" * 104)
    print("SWEEP SUMMARY: node RMS displacement error vs. h, l0_factor "
          "(valid = fair measure; recoverable over-counts the boundary shell)")
    print("=" * 104)
    print(f"{'h':>4} {'l0_factor':>10} {'l0(vox)':>8} {'rec.nodes':>12} "
          f"{'rec RMS':>10} {'rec max':>10} {'val.nodes':>12} {'valid RMS':>10} "
          f"{'valid max':>10} {'final dU/U':>12}  status")

    for r in results:
        flag = ''
        if r['status'] != 'ok':
            flag = r['status']
        elif r['rms'] > DIVERGED_RMS_THRESHOLD:
            flag = 'DIVERGED'

        rec_str = f"{r['n_recoverable']}/{r['n_total']}"
        val_str = f"{r['n_valid']}/{r['n_total']}"
        print(f"{r['h']:>4} {r['l0_factor']:>10g} {r['l0']:>8g} {rec_str:>12} "
              f"{r['rms']:>10.5f} {r['max_err']:>10.5f} {val_str:>12} "
              f"{r['rms_valid']:>10.5f} {r['max_err_valid']:>10.5f} "
              f"{r['du_final']:>12.2e}  {flag}")

    best_rec = _best(results, 'rms')
    best_val = _best(results, 'rms_valid')
    if best_rec is not None:
        print(f"\nBest by recoverable RMS (non-diverged): h={best_rec['h']}, "
              f"l0_factor={best_rec['l0_factor']:g} (l0={best_rec['l0']:g} vox) "
              f"-> RMS={best_rec['rms']:.5f} vox")
    if best_val is not None:
        print(f"Best by valid RMS (non-diverged, fair measure): h={best_val['h']}, "
              f"l0_factor={best_val['l0_factor']:g} (l0={best_val['l0']:g} vox) "
              f"-> RMS={best_val['rms_valid']:.5f} vox")


def write_csv(path, results):
    fieldnames = ['h', 'l0_factor', 'l0', 'n_recoverable', 'n_total',
                  'rms', 'max_err', 'n_valid', 'rms_valid', 'max_err_valid',
                  'du_final', 'status']
    with open(path, 'w', newline='') as fh:
        writer = csv_module.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved sweep summary CSV -> {path}")


def main(argv=None):
    args = parse_args(argv)
    h_values = [int(x) for x in args.h_values.split(',')]
    l0_factors = [float(x) for x in args.l0_factors.split(',')]
    output_root = os.path.abspath(args.output_root)

    results = run_sweep(h_values, l0_factors, args.gt_ref_file, args.gt_displacement_file,
                         args.tensile_strain, output_root)

    print_summary_table(results)
    if args.csv_path:
        write_csv(args.csv_path, results)


if __name__ == "__main__":
    main()
