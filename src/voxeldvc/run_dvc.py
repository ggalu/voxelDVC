# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-18 14:23:38
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-07-04 20:12:34

"""
This code is used after preprocessing an image pair with preprocess.py.
preprocess.py writes three files into a working directory WORK_DIR:
- ref_preprocessed.npy:  reference image cropped to the valid overlap region
- def_preprocessed.npy:  original deformed image cropped to the affine-mapped
                          bounding box [def_lo, def_hi); contains all deformed
                          samples needed for the reference crop without any
                          interpolation (used for DVC correlation)
- affine_prealign.json:   affine A, t, c, crop_lo/hi, def_lo/hi, bin_factor,
                          additional_crop_xyz, def_file (path to original deformed)

Correlation uses def_preprocessed (the affine-mapped crop) with a suitably
adjusted ext_affine, so exactly one trilinear interpolation is applied:
    c_adj  = c - crop_lo
    t_adj  = t + crop_lo - def_lo    (maps crop-ref voxel x to
             def_preprocessed index A@(x+crop_lo-c)+c+t-def_lo)

Because the DVC solver operates in def_preprocessed coordinate space, its
output U encodes displacements to that space.  Before saving, def_lo is
added back to each component so that U_recovered.npy represents the physical
displacement to full-image deformed coordinates -- identical to the output
that would be obtained by passing the full deformed image.

compute_active_mask in write_outputs receives g_origin=def_lo so it samples
def_preprocessed at (x + u_full - def_lo), correctly reproducing the
in-bounds check over the original deformed extent.

The reference-aligned deformed crop (deformed[crop_lo:crop_hi], same shape as
ref) is stored by preprocess.py as def_ref_aligned.npy and loaded here
directly; the original deformed file is not read at runtime.
"""


import argparse
import json
import os
import sys
import time

import numpy as np
import cupy as cp

from .engine.geometry_dvc import build_K_ref_laplacian
from .engine.correlate_gpu import multiscale_correlate_gpu
from .engine.write_output import Tee, write_outputs, mesh_dims
from .dvc_defaults import DEFAULT_L0_FACTOR
from ._format import kv_box, fmt_duration


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('work_dir', type=str)
    p.add_argument('--l0-factor', type=float, default=DEFAULT_L0_FACTOR, dest='l0_factor',
                   help="Regularization length l0 = l0_factor * h, in voxels "
                        f"(default: {DEFAULT_L0_FACTOR}).")
    return p.parse_args(argv)


def main(argv=None):
    t_start = time.perf_counter()

    ARGS = parse_args(argv)
    WORK_DIR = os.path.expanduser(ARGS.work_dir)

    log_path = os.path.join(WORK_DIR, 'run_log.txt')
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    # ---- load preprocessing metadata ----
    json_path = os.path.join(WORK_DIR, 'affine_prealign.json')
    with open(json_path) as fh:
        meta = json.load(fh)

    h        = meta['h']
    glt      = float(meta.get('glt', 0.0))   # gray-level threshold (old JSONs: 0.0)
    A_aff    = np.array(meta['affine_A'])
    t_aff    = np.array(meta['affine_t'])
    c_aff    = np.array(meta['affine_c'])
    crop_lo  = np.array(meta['crop_lo'])
    crop_hi  = np.array(meta['crop_hi'])
    lo_def   = np.array(meta['def_lo'])

    # Adjust affine for both the reference crop offset and the deformed crop
    # origin.  For a ref-crop voxel x_crop, the DVC samples def_preprocessed at:
    #   A @ (x_crop + crop_lo - c) + c + t - def_lo
    # which equals A @ (x_crop - c_adj) + c_adj + t_adj with:
    c_adj = c_aff - crop_lo
    t_adj = t_aff + crop_lo - lo_def

    # ---- load images ----
    ref      = np.load(os.path.join(WORK_DIR, 'ref_preprocessed.npy'))
    deformed = np.load(os.path.join(WORK_DIR, 'def_preprocessed.npy'))

    deformed_cropped = np.load(os.path.join(WORK_DIR, 'def_ref_aligned.npy'))

    Nx_e, Ny_e, Nz_e = mesh_dims(ref.shape, h)
    l0        = ARGS.l0_factor * h
    K_ref_lap = build_K_ref_laplacian()

    kv_box("Run configuration", [
        ("work_dir",         WORK_DIR),
        ("h (element size)", h),
        ("glt (gray thresh)", glt),
        ("l0_factor",        ARGS.l0_factor),
        ("l0 [vox]",         f"{l0:g}"),
        ("mesh (Nx,Ny,Nz)",  f"{Nx_e} x {Ny_e} x {Nz_e}"),
        ("ref shape",        str(ref.shape)),
        ("deformed shape",   str(deformed.shape)),
    ])

    print("\nGPU matrix-free DVC (multiscale, ext_affine U0, original deformed) …")
    cp.get_default_memory_pool().free_all_blocks()

    t_setup_done = time.perf_counter()
    U, res, dU = multiscale_correlate_gpu(
        ref, deformed, Nx_e, Ny_e, Nz_e, h, xp=cp,
        scales=(2, 1, 0),
        l0=l0,
        glt=glt,
        K_ref_laplacian=K_ref_lap,
        dynamic_mask=True,
        fft_prealign=False,
        prealign_affine=False,
        ext_affine=(A_aff, t_adj, c_adj),
        disp=True,
        freeze_mask_after=4
    )
    cp.cuda.Stream.null.synchronize()

    t_corr_done = time.perf_counter()

    U   = cp.asnumpy(U)
    res = cp.asnumpy(res)
    dU  = cp.asnumpy(dU)   # last GN update -> per-cell convergence field

    assert np.all(np.isfinite(U)), "Recovered displacement contains NaN/inf"

    pool = cp.get_default_memory_pool()
    print(f"\nGPU pool: {pool.total_bytes() / 1e6:.1f} MB")

    # Subtract the pre-computed affine contribution to obtain the small local
    # residual displacement U_local, which is what push_reference_forward needs
    # (sampling ref at x - u_total would take x outside the ref domain because
    # the affine translates by ~19 voxels in x and ~13 voxels in z).
    nbx, nby, nbz = Nx_e + 1, Ny_e + 1, Nz_e + 1
    ii, jj, kk = np.meshgrid(np.arange(nbx), np.arange(nby), np.arange(nbz),
                               indexing='ij')
    X_nodes = np.stack([ii, jj, kk], axis=-1).astype(np.float64) * h  # crop voxels
    u_aff_grid = (X_nodes - c_adj) @ A_aff.T + c_adj + t_adj - X_nodes
    U_local = np.empty_like(U)
    for c in range(3):
        U_local[c::3] = (U[c::3].reshape(nbx, nby, nbz, order='F')
                         - u_aff_grid[..., c]).flatten(order='F')

    # U currently encodes displacements into def_preprocessed coordinates.
    # Add def_lo back to each component so that U_recovered.npy represents the
    # physical displacement to full-image deformed coordinates, consistent with
    # the output of approaches that use the full deformed image.
    for c in range(3):
        U[c::3] += lo_def[c]

    # g_origin=lo_def tells compute_active_mask to subtract def_lo from the
    # sampled coordinates, correctly mapping U (now in full-image deformed
    # coordinates) into def_preprocessed indices.
    write_outputs(WORK_DIR, U, res, ref, deformed, deformed_cropped,
                  Nx_e, Ny_e, Nz_e, h, g_origin=tuple(int(x) for x in lo_def),
                  U_push=U_local, crop_lo=tuple(int(x) for x in crop_lo), glt=glt,
                  dU=dU)
    print("\nDVC COMPLETE")

    t_output_done = time.perf_counter()

    setup_s  = t_setup_done  - t_start
    corr_s   = t_corr_done   - t_setup_done
    output_s = t_output_done - t_corr_done
    total_s  = t_output_done - t_start
    kv_box("Timing", [
        ("Setup (load + mesh)",     fmt_duration(setup_s)),
        ("Multiscale correlation",  fmt_duration(corr_s)),
        ("Output generation",       fmt_duration(output_s)),
        ("Total",                   f"[bold]{fmt_duration(total_s)}[/bold]"),
    ], border_style="magenta", value_justify="right")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
