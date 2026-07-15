# -*- coding: utf-8 -*-
"""analyse_ground_truth.py: compare the DVC result produced by run_dvc.py
against the known ground-truth displacement field (stage 4 of 4; see
run_ground_truth_pipeline.py for the full chain: ground_truth/prepare_ground_truth.py ->
preprocess.py -> run_dvc.py -> this script).

Inputs
------
--work-dir DIR (default dvc_elastic_groundtruth): the directory
  preprocess.py and run_dvc.py wrote into. Must contain
  affine_prealign.json (for `h` and `crop_lo`), U_recovered.npy (nodal
  displacement) and ref.npy (the reference crop, used only for its shape via
  voxeldvc.engine.write_output.mesh_dims).

--gt-dir DIR (default dvc_elastic_groundtruth/gt_input): the directory
  prepare_ground_truth.py wrote into. Must contain u_gt_vox.npy (ground-truth
  displacement at every voxel of the FULL, uncropped reference image) and
  deformed.npy (the full, uncropped deformed image, used only for its shape
  to determine which voxels/nodes are recoverable).

Two coordinate-frame offsets have to be undone to compare U_recovered.npy to
the ground truth:

1. Mesh node (i, j, k) sits at FULL-image reference position
   `crop_lo + (i, j, k) * h`, not `(i, j, k) * h` -- accounted for by indexing
   into u_gt_vox.npy / deformed.npy from --gt-dir at `crop_lo`-shifted
   positions, rather than re-interpolating the ground-truth field (that
   interpolation happened once already, in
   ground_truth/prepare_ground_truth.py).
2. run_dvc.py's U_recovered.npy encodes displacement from the
   CROP-LOCAL reference index `(i,j,k)*h` to the full-image deformed
   position (only the deformed-side offset `def_lo` is added back there, see
   its module docstring) -- i.e. U_recovered = (full-image deformed
   position) - (crop-local reference position) = u_gt(x_full) + crop_lo.
   Subtracting `crop_lo` from U_recovered.npy right after loading converts
   it to displacement from the FULL-image reference position, matching the
   ground truth's convention, before any further use (both the nodal
   comparison and interp_field_vox's linear interpolation for the per-voxel
   comparison commute with this constant shift).

Known limitation: this assumes preprocess.py was run with --bin 1 (the
default) -- u_gt_vox.npy is defined on the unbinned voxel grid, so a binned
work-dir would need its crop coordinates rescaled before indexing into it.
The ground-truth pipeline does not use --bin, so this is not handled.

In addition to the nodal comparison, the recovered field is also evaluated at
every voxel of the reference crop (via the same trilinear FE shape functions
the solver uses, voxeldvc.engine.correlate_gpu.interp_field_vox) and compared against
the ground truth at voxel resolution. Both comparisons split out
voxels/nodes whose ground-truth-deformed position falls outside the deformed
image (unrecoverable/truncated) since their error reflects data missing from
g, not method error.

They additionally report a "valid" subset that, on top of the truncation
filter, drops the points the pipeline itself flags as untrustworthy on real
data via the saved solver masks (unsafe_voxel_mask.npy and
overlap_lost_element_mask.npy) -- chiefly the regularizer-biased outer
boundary shell (boundary_element_mask), plus any overlap-lost / low-material
elements. The "valid" RMS is therefore the accuracy measure consistent with
the masked output a real run would actually report on; "recoverable"
(truncation only) still over-counts the boundary shell. If those mask files
are absent (an older work_dir), the extra rows are silently skipped.
"""

import argparse
import json
import os
import sys

import numpy as np

from ..engine.geometry_dvc import build_N_stencil, build_node_index, element_node_ids
from ..engine.correlate_gpu import interp_field_vox
from ..engine.write_output import Tee, mesh_dims

# Default work/gt dirs are relative to the current working directory (there's
# no meaningful "repo root" once this package is installed). These apply only
# when this stage is run standalone; run_ground_truth_pipeline.py always
# passes an explicit --work-dir/--gt-dir (from its required --output-dir).
DEFAULT_WORK_DIR = os.path.abspath('dvc_elastic_groundtruth')
DEFAULT_GT_DIR = os.path.join(DEFAULT_WORK_DIR, 'gt_input')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work-dir', type=str, default=DEFAULT_WORK_DIR, dest='work_dir',
                   help=f"preprocess.py/run_dvc.py output directory (default: {DEFAULT_WORK_DIR})")
    p.add_argument('--gt-dir', type=str, default=DEFAULT_GT_DIR, dest='gt_dir',
                   help=f"ground_truth/prepare_ground_truth.py output directory (default: {DEFAULT_GT_DIR})")
    return p.parse_args(argv)


def grid_to_dofs(U_grid):
    """Inverse of the node-major DOF -> (nbx,nby,nbz,3) reshape used
    throughout this codebase (see voxeldvc.engine.write_output.write_outputs): flattens a
    (nbx,nby,nbz,3) nodal grid back into the flat interleaved DOF layout
    (U[3*node+c]) that interp_field_vox expects.
    """
    nbx, nby, nbz, _ = U_grid.shape
    n_nodes = nbx * nby * nbz
    U_flat = np.empty(n_nodes * 3, dtype=U_grid.dtype)
    for c in range(3):
        U_flat[c::3] = U_grid[..., c].flatten(order='F')
    return U_flat


def voxel_in_bounds(u_gt_vox, g_shape):
    """Per-voxel mask: True where the GT-deformed position x + u_gt(x) falls
    inside the deformed image g (same in-bounds test correlate_gpu's GN loop
    applies), i.e. where the voxel's ground-truth displacement is actually
    recoverable from the image pair rather than being invisible in g.

    u_gt_vox : (nx, ny, nz, 3) GT displacement, indexed in the same
        coordinate frame as g_shape (i.e. full-image coordinates when
        u_gt_vox is the full array).
    g_shape  : (gnx, gny, gnz) shape of the deformed image (g_origin assumed 0).
    """
    nx, ny, nz = u_gt_vox.shape[:3]
    gnx, gny, gnz = g_shape
    X, Y, Z = np.meshgrid(np.arange(nx, dtype=float),
                          np.arange(ny, dtype=float),
                          np.arange(nz, dtype=float), indexing='ij')
    cx = X + u_gt_vox[..., 0]
    cy = Y + u_gt_vox[..., 1]
    cz = Z + u_gt_vox[..., 2]
    return ((cx >= 0) & (cx <= gnx - 1) &
            (cy >= 0) & (cy <= gny - 1) &
            (cz >= 0) & (cz <= gnz - 1))


def node_oob_fraction(u_gt_vox_full, h, nbfx, nbfy, nbfz, g_shape, crop_lo):
    """Per-node fraction of supporting voxels that map outside the deformed
    image under the ground-truth displacement -- i.e. the truncation that the
    DVC cannot recover.

    Mesh node (i, j, k) sits at full-image reference position
    crop_lo + (i, j, k) * h; its support is the (up to 8) elements sharing
    it, i.e. the voxel box [v-h, v+h] (clamped to the full image) around that
    position. The returned (nbfx, nbfy, nbfz) array gives, per node, the
    fraction of that support box that is out of bounds; a node with fraction
    0 is fully recoverable, fraction > 0 is "truncated".

    u_gt_vox_full : (nx, ny, nz, 3) GT displacement at every voxel of the
        FULL (uncropped) reference image.
    h        : mesh element edge length (voxels).
    nbfx, nbfy, nbfz : nodes per axis (Nx_e+1, Ny_e+1, Nz_e+1) -- the mesh
        need not be cubic (preprocess.py's overlap crop can clip axes
        differently).
    g_shape  : (gnx, gny, gnz) shape of the full deformed image.
    crop_lo  : (3,) int -- reference-crop origin in full-image coordinates.
    """
    nx, ny, nz = u_gt_vox_full.shape[:3]
    in_bounds = voxel_in_bounds(u_gt_vox_full, g_shape)

    oob_frac = np.zeros((nbfx, nbfy, nbfz))
    for i in range(nbfx):
        for j in range(nbfy):
            for k in range(nbfz):
                vi, vj, vk = crop_lo[0] + i * h, crop_lo[1] + j * h, crop_lo[2] + k * h
                sx = slice(max(vi - h, 0), min(vi + h, nx - 1) + 1)
                sy = slice(max(vj - h, 0), min(vj + h, ny - 1) + 1)
                sz = slice(max(vk - h, 0), min(vk + h, nz - 1) + 1)
                oob_frac[i, j, k] = 1.0 - in_bounds[sx, sy, sz].mean()
    return oob_frac


def load_validity_masks(work_dir, Nx_e, Ny_e, Nz_e):
    """Load the solver-side validity masks saved by write_outputs and derive
    per-node / per-voxel "masked out" flags, so the ground-truth accuracy
    statistics can be restricted to the same trustworthy set the pipeline
    reports on real (unknown-GT) data -- not just the truncation-recoverable
    set. Returns (node_masked_out (nbx,nby,nbz) bool, vox_masked_out
    (Nvx,Nvy,Nvz) bool) or (None, None) if the masks are absent (e.g. a
    work_dir from an older run_dvc that predates them).

    - vox_masked_out is `unsafe_voxel_mask.npy` directly (per-voxel: inactive-
      voxel dilation + outer boundary shell).
    - node_masked_out dilates `overlap_lost_element_mask.npy` (the porosity-
      aware cell-strain gate, which now also flags the boundary shell) from
      elements onto nodes: a node is masked out if ANY element touching it is
      flagged. Both masks include the regularizer-biased boundary shell
      (boundary_element_mask), so the "valid" set excludes it.
    """
    elem_path = os.path.join(work_dir, 'overlap_lost_element_mask.npy')
    vox_path = os.path.join(work_dir, 'unsafe_voxel_mask.npy')
    if not (os.path.exists(elem_path) and os.path.exists(vox_path)):
        return None, None

    vox_masked_out = np.load(vox_path).astype(bool)               # (Nvx,Nvy,Nvz)

    overlap_lost = np.load(elem_path).astype(bool)                # (Nx_e,Ny_e,Nz_e)
    # overlap_lost[i,j,k] was saved with order='F', i.e. elem_id = i + j*Nx +
    # k*Nx*Ny is recovered by an order='F' flatten -- matching element_node_ids.
    ol_flat = overlap_lost.flatten(order='F')                     # (Nelem,) x-fastest
    node_idx = build_node_index(Nx_e, Ny_e, Nz_e)                 # (nbx,nby,nbz)
    elem_nodes = element_node_ids(node_idx, Nx_e, Ny_e, Nz_e)     # (Nelem, 8)
    node_masked_flat = np.zeros(node_idx.size, dtype=bool)
    node_masked_flat[elem_nodes[ol_flat].ravel()] = True
    node_masked_out = node_masked_flat[node_idx]                  # (nbx,nby,nbz)
    return node_masked_out, vox_masked_out


def main(argv=None):
    args = parse_args(argv)
    work_dir = os.path.expanduser(args.work_dir)
    gt_dir = os.path.expanduser(args.gt_dir)

    log_path = os.path.join(work_dir, 'ground_truth_comparison_log.txt')
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 70)
    print("GROUND-TRUTH COMPARISON")
    print("=" * 70)

    # --- load preprocessing metadata + DVC result ---
    with open(os.path.join(work_dir, 'affine_prealign.json')) as fh:
        meta = json.load(fh)
    h = meta['h']
    crop_lo = np.array(meta['crop_lo'])

    # U_recovered.npy encodes displacement from the crop-local reference index
    # to the full-image deformed position; subtract crop_lo to get displacement
    # from the full-image reference position, matching the GT convention (see
    # module docstring point 2).
    U_recovered_grid = np.load(os.path.join(work_dir, 'U_recovered.npy')) - crop_lo  # (nbx,nby,nbz,3)
    ref_crop = np.load(os.path.join(work_dir, 'ref.npy'))
    Nx_e, Ny_e, Nz_e = mesh_dims(ref_crop.shape, h)
    nbfx, nbfy, nbfz = Nx_e + 1, Ny_e + 1, Nz_e + 1
    assert U_recovered_grid.shape[:3] == (nbfx, nbfy, nbfz)
    print(f"work_dir={work_dir}  mesh: Nx_e={Nx_e}, Ny_e={Ny_e}, Nz_e={Nz_e}  h={h}  "
          f"crop_lo={crop_lo.tolist()}")

    # --- load ground truth ---
    u_gt_vox_full = np.load(os.path.join(gt_dir, 'u_gt_vox.npy'))  # full-image, (nx,ny,nz,3)
    deformed_full = np.load(os.path.join(gt_dir, 'deformed.npy'))
    print(f"gt_dir={gt_dir}  u_gt_vox_full shape={u_gt_vox_full.shape}  "
          f"deformed_full shape={deformed_full.shape}")

    # --- ground-truth comparison at mesh nodes ---
    # Node (i, j, k) sits at full-image reference position crop_lo + (i,j,k)*h.
    gx = crop_lo[0] + np.arange(nbfx) * h
    gy = crop_lo[1] + np.arange(nbfy) * h
    gz = crop_lo[2] + np.arange(nbfz) * h
    u_gt_nodes = u_gt_vox_full[np.ix_(gx, gy, gz)]  # (nbfx,nbfy,nbfz,3)

    error = U_recovered_grid - u_gt_nodes

    # --- truncation mask: nodes whose support partly maps outside g ---
    oob_frac = node_oob_fraction(u_gt_vox_full, h, nbfx, nbfy, nbfz, deformed_full.shape, crop_lo)
    truncated = oob_frac > 0           # (nbfx,nbfy,nbfz) bool
    recoverable = ~truncated

    # Solver-side validity masks (boundary shell + overlap/unsafe), saved by
    # write_outputs. "recoverable" only removes truncation (data missing from
    # g); the "valid" set additionally removes points the pipeline itself
    # flags as untrustworthy on real data -- chiefly the regularizer-biased
    # boundary shell -- so it is the accuracy measure consistent with the
    # masked output a real run would report on. May be absent for an old
    # work_dir, in which case the extra rows are skipped.
    node_masked_out, vox_masked_out = load_validity_masks(work_dir, Nx_e, Ny_e, Nz_e)
    have_masks = node_masked_out is not None
    node_valid = recoverable & ~node_masked_out if have_masks else None

    print("\n" + "=" * 70)
    print("GROUND-TRUTH COMPARISON (at mesh nodes)")
    print("=" * 70)
    print(f"{'Component':<12} {'GT mean':>10} {'GT std':>10} "
          f"{'Rec mean':>10} {'Rec std':>10} {'RMS err':>10} {'Max |err|':>10}")
    for c, name in enumerate(['ux', 'uy', 'uz']):
        gt_c = u_gt_nodes[..., c]
        rec_c = U_recovered_grid[..., c]
        err_c = error[..., c]
        rms = np.sqrt(np.mean(err_c**2))
        print(f"  {name:<10} {gt_c.mean():>10.5f} {gt_c.std():>10.5f} "
              f"{rec_c.mean():>10.5f} {rec_c.std():>10.5f} "
              f"{rms:>10.5f} {np.max(np.abs(err_c)):>10.5f}")

    def region_stats(mask):
        if not np.any(mask):
            return 0, float('nan'), float('nan')
        err_m = error[mask]                       # (n, 3)
        rms = np.sqrt(np.mean(err_m**2))          # over all 3 components
        max_abs = np.max(np.linalg.norm(err_m, axis=-1))
        return int(mask.sum()), rms, max_abs

    print("\n  Displacement error by region (truncated nodes reported separately):")
    print(f"  {'Region':<34} {'#nodes':>7} {'RMS err':>10} {'Max |err|':>10}")
    regions = [('recoverable (support in g)', recoverable)]
    if have_masks:
        regions.append(('valid (recoverable & unmasked)', node_valid))
    regions += [
        ('truncated (support clipped)', truncated),
        ('all nodes', np.ones_like(truncated)),
    ]
    for label, mask in regions:
        n, rms, max_abs = region_stats(mask)
        print(f"  {label:<34} {n:>7d} {rms:>10.5f} {max_abs:>10.5f}")
    print(f"\n  Truncated nodes: {int(truncated.sum())}/{truncated.size} "
          f"(max support out-of-bounds fraction = {oob_frac.max():.3f})")
    if have_masks:
        print(f"  Masked-out (boundary shell / overlap-lost) among recoverable: "
              f"{int((recoverable & node_masked_out).sum())}/{int(recoverable.sum())}")
        print("  -> the 'valid' RMS/max is the fair measure of DVC accuracy: it drops")
        print("     both truncated nodes (data missing from g) AND nodes the pipeline")
        print("     flags untrustworthy (regularizer-biased boundary shell, overlap loss).")
    else:
        print("  -> the 'recoverable' RMS/max is the fair measure of DVC accuracy;")
        print("     truncated-node error is dominated by data missing from g, not method error.")

    np.save(os.path.join(work_dir, 'U_gt_nodes.npy'), u_gt_nodes)
    np.save(os.path.join(work_dir, 'U_error.npy'), error)
    np.save(os.path.join(work_dir, 'node_truncated_mask.npy'), truncated)
    np.save(os.path.join(work_dir, 'node_oob_fraction.npy'), oob_frac)
    if have_masks:
        np.save(os.path.join(work_dir, 'node_valid_mask.npy'), node_valid)
    print(f"\nSaved U_gt_nodes.npy {u_gt_nodes.shape}, U_error.npy {error.shape}, "
          f"node_truncated_mask.npy {truncated.shape} and "
          f"node_oob_fraction.npy {oob_frac.shape} -> {work_dir}")

    # --- ground-truth comparison at every voxel of the reference crop ---
    Nvx, Nvy, Nvz = ref_crop.shape
    u_gt_vox = u_gt_vox_full[crop_lo[0]:crop_lo[0] + Nvx,
                              crop_lo[1]:crop_lo[1] + Nvy,
                              crop_lo[2]:crop_lo[2] + Nvz]

    N_stencil, _ = build_N_stencil(h)
    U_flat = grid_to_dofs(U_recovered_grid)
    u_rec_vox = interp_field_vox(U_flat, N_stencil, h, Nx_e, Ny_e, Nz_e, np)
    u_rec_vox = u_rec_vox.reshape(Nvx, Nvy, Nvz, 3, order='F')
    error_vox = u_rec_vox - u_gt_vox

    vox_recoverable = voxel_in_bounds(u_gt_vox_full, deformed_full.shape)[
        crop_lo[0]:crop_lo[0] + Nvx,
        crop_lo[1]:crop_lo[1] + Nvy,
        crop_lo[2]:crop_lo[2] + Nvz]

    def print_voxel_table(sel):
        for c, name in enumerate(['ux', 'uy', 'uz']):
            gt_c = u_gt_vox[..., c][sel]
            rec_c = u_rec_vox[..., c][sel]
            err_c = error_vox[..., c][sel]
            rms = np.sqrt(np.mean(err_c**2))
            print(f"  {name:<10} {gt_c.mean():>10.5f} {gt_c.std():>10.5f} "
                  f"{rec_c.mean():>10.5f} {rec_c.std():>10.5f} "
                  f"{rms:>10.5f} {np.max(np.abs(err_c)):>10.5f}")

    print("\n" + "=" * 70)
    print("GROUND-TRUTH COMPARISON (at every voxel)")
    print("=" * 70)
    header = (f"{'Component':<12} {'GT mean':>10} {'GT std':>10} "
              f"{'Rec mean':>10} {'Rec std':>10} {'RMS err':>10} {'Max |err|':>10}")
    print(header)
    print_voxel_table(vox_recoverable)
    print(f"  (restricted to recoverable voxels: {int(vox_recoverable.sum())} "
          f"-- support in g under the GT field)")

    if have_masks:
        vox_valid = vox_recoverable & ~vox_masked_out
        print("\n" + header)
        print_voxel_table(vox_valid)
        print(f"  (restricted to VALID voxels: {int(vox_valid.sum())} -- recoverable AND "
              f"not mask-flagged;")
        print("   drops the regularizer-biased boundary shell + overlap-lost voxels)")

    np.save(os.path.join(work_dir, 'U_gt_vox.npy'), u_gt_vox)
    np.save(os.path.join(work_dir, 'U_recovered_vox.npy'), u_rec_vox)
    np.save(os.path.join(work_dir, 'U_error_vox.npy'), error_vox)
    np.save(os.path.join(work_dir, 'voxel_recoverable_mask.npy'), vox_recoverable)
    if have_masks:
        np.save(os.path.join(work_dir, 'voxel_valid_mask.npy'),
                vox_recoverable & ~vox_masked_out)
    print(f"\nSaved U_gt_vox.npy {u_gt_vox.shape}, U_recovered_vox.npy {u_rec_vox.shape}, "
          f"U_error_vox.npy {error_vox.shape} and voxel_recoverable_mask.npy "
          f"{vox_recoverable.shape} -> {work_dir}")

    print("\n✓ GROUND-TRUTH COMPARISON COMPLETE")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
