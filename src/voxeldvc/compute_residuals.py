#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_residuals.py

Standalone post-processing script: recomputes the ZNCC-normalised DVC
residual field from the output files written by voxeldvc.engine.write_output.write_outputs
and reports the same residual norm (std(res) in grey-level units) that
correlate_gpu prints during the Gauss-Newton iteration.

The residual at a reference voxel x is

    res(x) = [f(x) - mean0] - (std0/std1) * [g(x + u(x)) - g_mean]

where f=ref.npy, u=FE-interpolated U_recovered.npy, and g is sampled from
the deformed image.  Two deformed-image sources are tried in order:

  1. def_preprocessed.npy  (preferred -- the actual DVC input)
     Coordinate mapping:  g_coord = x + U_recovered - def_lo
     This exactly reproduces the GPU's active-voxel mask and residual.

  2. deformed.npy  (fallback -- reference-indexed deformed crop)
     Coordinate mapping:  g_coord = x + U_recovered - crop_lo
     Boundary voxels displaced by the affine translation fall outside this
     crop and are counted inactive, so the active fraction is lower than
     the GPU reported.

mean0/std0 are the quadrature-point statistics of f over material voxels;
g_mean/std1 are the same for g over active voxels.  std(res) is computed
over the quadrature-point array (matching the GPU's "std(res)=X.XX gl").

Usage
-----
  python compute_residuals.py <work_dir>

  work_dir must contain:
    ref.npy              (Nvx, Nvy, Nvz)  reference image
    U_recovered.npy      (Nx+1, Ny+1, Nz+1, 3)  nodal displacement field
    affine_prealign.json  def_lo / crop_lo for coordinate mapping
    residual.npy         (Nvx, Nvy, Nvz)  stored residual (for comparison)

  and either:
    def_preprocessed.npy  (preferred)  or  deformed.npy  (fallback)
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates


# ------------------------------------------------------------------ helpers --

def gather_qp(field, h, Nx, Ny, Nz):
    """Replicate a (Nvx,Nvy,Nvz) field at quadrature points.

    Returns a flat ((Nx*Ny*Nz)*(h+1)^3,) array with the same element/stencil
    ordering as correlate_gpu's gather_quadpoint_values.  Interior voxels
    appear in multiple elements; boundary voxels appear fewer times.
    The input field is Fortran-indexed (x fastest).
    """
    Nvx = Nx * h + 1
    Nvy = Ny * h + 1
    q = np.arange(h + 1, dtype=np.int64)
    qx, qy, qz = np.meshgrid(q, q, q, indexing='ij')
    offsets = (qx + qy * Nvx + qz * Nvx * Nvy).ravel()  # (h+1)^3

    Ne = Nx * Ny * Nz
    ie = np.arange(Ne, dtype=np.int64)
    i_e = ie % Nx
    j_e = (ie // Nx) % Ny
    k_e = ie // (Nx * Ny)
    vox0 = h * (i_e + j_e * Nvx + k_e * Nvx * Nvy)

    vox_ids = (vox0[:, None] + offsets[None, :]).ravel()
    return field.ravel(order='F')[vox_ids]


def interp_u_to_voxels(U_recovered, h):
    """Trilinearly interpolate nodal U_recovered (Nx+1,Ny+1,Nz+1,3) to
    every voxel centre, returning a (3, Nvox) float32 array."""
    nbx, nby, nbz, _ = U_recovered.shape
    Nx, Ny, Nz = nbx - 1, nby - 1, nbz - 1
    Nvx = Nx * h + 1
    Nvy = Ny * h + 1
    Nvz = Nz * h + 1

    # Fractional node-grid coordinates for every voxel
    vx = np.arange(Nvx, dtype=np.float32) / h
    vy = np.arange(Nvy, dtype=np.float32) / h
    vz = np.arange(Nvz, dtype=np.float32) / h
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')
    node_coords = np.array([VX.ravel(), VY.ravel(), VZ.ravel()])  # (3, Nvox)
    del VX, VY, VZ

    u = np.empty((3, Nvx * Nvy * Nvz), dtype=np.float32)
    for c in range(3):
        u[c] = map_coordinates(U_recovered[..., c].astype(np.float32),
                               node_coords, order=1, mode='nearest')
    return u, (Nvx, Nvy, Nvz)


# -------------------------------------------------------------------- main --

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('work_dir', help='Directory containing DVC output files')
    args = parser.parse_args(argv)

    d = args.work_dir

    # ---- load required files ----
    ref        = np.load(os.path.join(d, 'ref.npy'))
    U_nodal    = np.load(os.path.join(d, 'U_recovered.npy'))   # (Nx+1,Ny+1,Nz+1,3)
    res_stored = np.load(os.path.join(d, 'residual.npy'))      # (Nvx,Nvy,Nvz)

    json_path = os.path.join(d, 'affine_prealign.json')
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found -- needed for def_lo / crop_lo", file=sys.stderr)
        sys.exit(1)
    with open(json_path) as fh:
        meta = json.load(fh)

    h = int(meta['h'])
    glt = float(meta.get('glt', 0.0))   # gray-level threshold (old JSONs: 0.0)

    # Prefer def_preprocessed.npy (exact GPU match); fall back to deformed.npy.
    def_pre_path = os.path.join(d, 'def_preprocessed.npy')
    def_crop_path = os.path.join(d, 'deformed.npy')
    if os.path.exists(def_pre_path):
        deformed = np.load(def_pre_path)
        g_offset = np.array(meta['def_lo'])   # U_recovered = U_DVC + def_lo → g_coord = x + U_recovered - def_lo
        print(f"Deformed source : def_preprocessed.npy  (exact GPU match)")
    elif os.path.exists(def_crop_path):
        deformed = np.load(def_crop_path)
        g_offset = np.array(meta['crop_lo'])
        print(f"Deformed source : deformed.npy  (fallback; boundary voxels may differ from GPU)")
    else:
        print("ERROR: neither def_preprocessed.npy nor deformed.npy found", file=sys.stderr)
        sys.exit(1)

    Nvx, Nvy, Nvz = ref.shape
    nbx, nby, nbz = U_nodal.shape[:3]
    Nx, Ny, Nz = nbx - 1, nby - 1, nbz - 1

    # Sanity check: shapes must be consistent with h
    assert Nvx == Nx * h + 1 and Nvy == Ny * h + 1 and Nvz == Nz * h + 1, (
        f"Shape mismatch: ref={ref.shape}, U_nodal[:3]={U_nodal.shape[:3]}, h={h}")
    # def_preprocessed may have a different shape than ref (larger affine bbox)

    print(f"Work directory : {d}")
    print(f"Mesh           : Nx={Nx}, Ny={Ny}, Nz={Nz},  h={h}")
    print(f"Image shape    : {ref.shape}  ({Nvx*Nvy*Nvz:,} voxels)\n")

    # ------------------------------------------------------------------ recompute --
    print("Interpolating displacement to voxel centres …")
    u_vox, _ = interp_u_to_voxels(U_nodal, h)   # (3, Nvox) float32

    # Deformed-image coordinates: voxel position + u(x) - g_offset
    vx = np.arange(Nvx, dtype=np.float32)
    vy = np.arange(Nvy, dtype=np.float32)
    vz = np.arange(Nvz, dtype=np.float32)
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')   # (Nvx,Nvy,Nvz) each

    g_coords = np.array([
        VX.ravel() + u_vox[0] - g_offset[0],
        VY.ravel() + u_vox[1] - g_offset[1],
        VZ.ravel() + u_vox[2] - g_offset[2],
    ], dtype=np.float32)
    del VX, VY, VZ, u_vox

    Gx, Gy, Gz = deformed.shape
    in_bounds = (
        (g_coords[0] >= 0) & (g_coords[0] <= Gx - 1) &
        (g_coords[1] >= 0) & (g_coords[1] <= Gy - 1) &
        (g_coords[2] >= 0) & (g_coords[2] <= Gz - 1)
    )

    print("Sampling deformed image at x + u(x) …")
    g_sampled = map_coordinates(
        deformed.astype(np.float32), g_coords,
        order=1, mode='constant', cval=0.0
    ).reshape(Nvx, Nvy, Nvz)
    del g_coords

    mask0  = (ref > glt)                       # reference material
    active = in_bounds.reshape(Nvx, Nvy, Nvz) & mask0 & (g_sampled > glt)
    del in_bounds

    n_active = int(active.sum())
    print(f"Active voxels  : {n_active}/{ref.size} ({100*n_active/ref.size:.1f}%)\n")

    # ---- ZNCC normalisation over quadrature points (exact GPU match) ----
    print("Computing ZNCC normalisation over quadrature points …")

    f_float = ref.astype(np.float32)
    g_float = g_sampled.astype(np.float32)
    g_float[~active] = 0.0

    mask0_qp  = gather_qp(mask0.astype(np.float32), h, Nx, Ny, Nz) > 0.5
    active_qp = gather_qp(active.astype(np.float32), h, Nx, Ny, Nz) > 0.5
    f_qp      = gather_qp(f_float, h, Nx, Ny, Nz)
    g_qp      = gather_qp(g_float, h, Nx, Ny, Nz)

    mean0 = float(f_qp[mask0_qp].mean())
    std0  = float(f_qp[mask0_qp].std())
    g_mean = float(g_qp[active_qp].mean())
    std1   = float((g_qp[active_qp] - g_mean).std())

    print(f"  f  : mean0={mean0:.2f} gl,  std0={std0:.2f} gl")
    print(f"  g  : g_mean={g_mean:.2f} gl,  std1={std1:.2f} gl")
    print(f"  std0/std1 = {std0/std1:.4f}\n")

    # ---- residual field ----
    f_flat = f_float.ravel(order='F')
    g_flat = g_float.ravel(order='F')
    act_flat = active.ravel(order='F')

    res_flat = np.zeros(Nvx * Nvy * Nvz, dtype=np.float32)
    res_flat[act_flat] = (
        (f_flat[act_flat] - mean0) -
        (std0 / std1) * (g_flat[act_flat] - g_mean)
    )
    res_recomputed = res_flat.reshape(Nvx, Nvy, Nvz, order='F')

    # ---- residual norm (QP-weighted, matching GPU printout) ----
    res_qp = gather_qp(res_flat.reshape(Nvx, Nvy, Nvz, order='F').astype(np.float32),
                       h, Nx, Ny, Nz)
    std_qp_recomputed = float(res_qp[active_qp].std())

    # Per-voxel std (simpler, slightly different due to multiplicity weighting)
    std_vox_recomputed = float(res_flat[act_flat].std())

    print("=" * 60)
    print("RECOMPUTED RESIDUAL")
    print("=" * 60)
    print(f"  std(res) QP-weighted   = {std_qp_recomputed:.4f} gl  "
          f"[matches GPU 'std(res)=X.XX gl' output]")
    print(f"  std(res) per-voxel     = {std_vox_recomputed:.4f} gl")
    print(f"  mean(res) [active]     = {float(res_flat[act_flat].mean()):.4f} gl")
    print(f"  max|res|  [active]     = {float(np.abs(res_flat[act_flat]).max()):.4f} gl")

    # ------------------------------------------------------------------ stored residual --
    print()
    print("=" * 60)
    print("STORED residual.npy  (cross-check)")
    print("=" * 60)

    stored_flat  = res_stored.ravel(order='F')
    stored_act   = stored_flat != 0
    stored_qp    = gather_qp(res_stored, h, Nx, Ny, Nz)
    stored_act_qp = stored_qp != 0

    n_stored_active = int(stored_act.sum())
    std_qp_stored  = float(stored_qp[stored_act_qp].std())
    std_vox_stored = float(stored_flat[stored_act].std())

    print(f"  Active voxels          : {n_stored_active}/{ref.size} "
          f"({100*n_stored_active/ref.size:.1f}%)")
    print(f"  std(res) QP-weighted   = {std_qp_stored:.4f} gl")
    print(f"  std(res) per-voxel     = {std_vox_stored:.4f} gl")
    print(f"  mean(res) [active]     = {float(stored_flat[stored_act].mean()):.4f} gl")
    print(f"  max|res|  [active]     = {float(np.abs(stored_flat[stored_act]).max()):.4f} gl")

    # ------------------------------------------------------------------ diff --
    diff = res_recomputed.astype(np.float64) - res_stored.astype(np.float64)
    act_either = act_flat | stored_act
    print()
    print("=" * 60)
    print("DIFFERENCE  (recomputed - stored)")
    print("=" * 60)
    print(f"  max|diff| over active voxels : {np.abs(diff.ravel(order='F')[act_either]).max():.4f} gl")
    print(f"  rms diff  over active voxels : {np.sqrt((diff.ravel(order='F')[act_either]**2).mean()):.4f} gl")

    # ------------------------------------------------------------------ save recomputed --
    out_path = os.path.join(d, 'residual_recomputed.npy')
    np.save(out_path, res_recomputed)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
