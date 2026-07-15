# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-18 22:29:59
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-28 17:56:37
#!/usr/bin/env python3
"""
crop_overlap_affine_MR4.py

1. Estimate affine alignment (rotation + per-axis scale + translation) between
   the reference and deformed CT volumes using affine_prealign.
2. Warp the deformed image onto the reference voxel grid (internally only) to
   find the valid overlap region.
3. Find the valid overlap: voxels where both the reference and the warped
   deformed are non-zero.
4. Crop both to the AABB of that region, snapped to n*h+1 per axis (h = DVC
   element edge length, default 8).
5. Save the cropped images and a JSON file with the full affine parameters.

def_preprocessed.npy contains the ORIGINAL (non-warped) deformed image
cropped to the same region as ref_preprocessed.npy (reference crop indices
applied directly to the deformed volume).  The affine alignment is not
baked into this file; it is stored in affine_prealign.json and applied by
run_dvc.py as the ext_affine initial displacement, avoiding the
double-interpolation artifact that would result from correlating against a
pre-warped image.

Usage
-----
  python preprocess.py <ref_path> <def_path> --output-dir DIR [--h H] [--bin B]

  ref_path       Path to reference image (.npy file)
  def_path       Path to deformed image (.npy file)
  --output-dir   Output directory (required)
  --h H          DVC element edge length in voxels (default: 8).
                 Output image sizes will satisfy  size = n*H + 1  per axis.
  --bin B        Spatial binning factor (default: 1 = no binning).
                 Each output voxel is the mean of B^3 input voxels.

Output
------
  ref_preprocessed.npy           reference cropped to the valid overlap region  original dtype  shape (n0*H+1, n1*H+1, n2*H+1)
  def_preprocessed.npy           deformed cropped to the affine-mapped bounding box [def_lo, def_hi)  original dtype
  def_ref_aligned.npy            deformed cropped to the same index range as the reference [crop_lo, crop_hi)  original dtype  shape = ref_preprocessed.npy shape
  affine_prealign.json           affine parameters + all crop metadata for post-processing

JSON content
------------
The JSON stores everything needed to reconstruct the total displacement from
the DVC residual field.  For a DVC node at cropped position x_crop the affine
displacement is:

    u_affine(x_crop) = (A - I) @ (x_crop + crop_lo - c) + t

and the total displacement is u_affine + u_dvc, where u_dvc comes from the
DVC solver run on the cropped image pair.
"""
import argparse
import json
import os
import sys
import numpy as np
import cupy as cp
import tifffile

from .engine.correlate_gpu import affine_prealign, affine_refine_gn
from .dvc_defaults import DEFAULT_H


# ------------------------------------------------------------------ helpers --

def load_volume(filepath):
    """Load volume from .npy or .tif/.tiff file.

    Returns
    -------
    vol : ndarray
    """
    _, ext = os.path.splitext(filepath)
    if ext.lower() == '.npy':
        return np.load(filepath)
    elif ext.lower() in ['.tif', '.tiff']:
        return tifffile.imread(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def print_params_box(params):
    """Pretty-print the parsed input parameters in a boxed key/value table.

    params : list of (label, value) tuples, printed in order.
    """
    from ._format import kv_box
    kv_box("Parameters as understood by the program", params)


def print_affine_box(A, angles, scales, t, c):
    """Pretty-print the recovered affine transform in a boxed table."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    tbl = Table(box=None, show_header=True, header_style="bold", pad_edge=False)
    tbl.add_column("",  style="bold cyan", justify="left")
    tbl.add_column("x", justify="right")
    tbl.add_column("y", justify="right")
    tbl.add_column("z", justify="right")
    # Row labels are built as Text() so unit brackets like [deg]/[vox] are not
    # parsed as rich markup tags (which would silently strip them).
    eps = normal_strains(A)
    tbl.add_row(Text("Rotation  [deg]"),   f"{angles[0]:+.3f}", f"{angles[1]:+.3f}", f"{angles[2]:+.3f}")
    tbl.add_row(Text("Scale"),             f"{scales[0]:.4f}",  f"{scales[1]:.4f}",  f"{scales[2]:.4f}")
    tbl.add_row(Text("Normal strain"),     f"{eps[0]:+.5f}",    f"{eps[1]:+.5f}",    f"{eps[2]:+.5f}")
    tbl.add_row(Text("Translation [vox]"), f"{t[0]:+.3f}",      f"{t[1]:+.3f}",      f"{t[2]:+.3f}")
    tbl.add_row(Text("Center [vox]"),      f"{c[0]:.2f}",       f"{c[1]:.2f}",       f"{c[2]:.2f}")

    Console().print(Panel(tbl,
                          title="[bold]Recovered affine transform[/bold]",
                          border_style="cyan", box=box.ROUNDED,
                          padding=(1, 3), expand=False))


def print_cost_box(info):
    """Pretty-print how the alignment residual improved stage-by-stage.

    The three costs are the normalized correlation residual (lower = better
    fit) at successive alignment stages: the raw identity transform, after
    the FFT translation-only estimate, and after the full affine fit.  The
    Improvement column is the residual reduction relative to the unaligned
    (identity) baseline.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    c_id = info['cost_identity']
    c_tr = info['cost_translation_only']
    c_af = info['cost_final']

    def improved(c):
        if c_id == 0:
            return "—"
        return f"{100.0 * (c_id - c) / c_id:+.1f}%"

    tbl = Table(box=None, show_header=True, header_style="bold", pad_edge=False)
    tbl.add_column("Alignment stage", style="bold cyan", justify="left")
    tbl.add_column("Residual", justify="right")
    tbl.add_column("vs. unaligned", justify="right")
    tbl.add_row("No alignment (identity)", f"{c_id:.4f}", "[dim]baseline[/dim]")
    tbl.add_row("+ translation (FFT)",     f"{c_tr:.4f}", improved(c_tr))
    tbl.add_row("+ full affine",           f"[green]{c_af:.4f}[/green]", f"[green]{improved(c_af)}[/green]")

    footer = f"[dim]lower residual = better fit · optimiser evaluations (nfev): {info['nfev']}[/dim]"

    from rich.console import Group
    Console().print(Panel(Group(tbl, "", footer),
                          title="[bold]Alignment quality[/bold]",
                          border_style="magenta", box=box.ROUNDED,
                          padding=(1, 3), expand=False))


def print_refine_box(info):
    """Pretty-print the Gauss-Newton affine refinement result.

    Reports the masked RMS intensity residual (def - ref warped by the
    affine) before and after the full-resolution Gauss-Newton refinement;
    a lower residual means the affine explains more of the deformation.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    r0 = info['rms_initial']
    r1 = info['rms_final']
    drop = "—" if r0 == 0 else f"{100.0 * (r0 - r1) / r0:+.1f}%"

    tbl = Table(box=None, show_header=True, header_style="bold", pad_edge=False)
    tbl.add_column("Stage", style="bold cyan", justify="left")
    tbl.add_column("RMS residual", justify="right")
    tbl.add_column("vs. prealign", justify="right")
    tbl.add_row("Coarse prealign", f"{r0:.2f}", "[dim]baseline[/dim]")
    tbl.add_row("+ Gauss-Newton refine", f"[green]{r1:.2f}[/green]",
                f"[green]{drop}[/green]")

    footer = (f"[dim]lower residual = better fit · GN iters/level: "
              f"{info['iters']}[/dim]")

    from rich.console import Group
    Console().print(Panel(Group(tbl, "", footer),
                          title="[bold]Affine refinement (Gauss-Newton)[/bold]",
                          border_style="green", box=box.ROUNDED,
                          padding=(1, 3), expand=False))


def rotation_angles_deg(A):
    """Decompose A into Euler rotation angles (deg, xyz) and per-axis stretch.

    The stretch returned is the DIAGONAL of the right stretch tensor P
    (A = R P polar decomposition), which keeps its x/y/z correspondence.
    Do NOT use the raw SVD singular values here: numpy returns them sorted
    descending, so for an anisotropic A they get mis-assigned to axes (e.g. a
    z-only stretch would be reported under x), which silently hides which
    axis actually deformed.
    """
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = R.copy()
        R[:, -1] *= -1
    rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    ry = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    # Right stretch tensor P = R^T A; its diagonal is the axis-aligned scale.
    scales = np.diag(R.T @ A).copy()
    return (rx, ry, rz), scales


def normal_strains(A):
    """Per-axis normal strain of the affine, in the same (physical, ref-frame)
    convention recover_affine.py reports: eps = diag(M) where the deformation
    is def(x) = ref(x - M(x-c) - t) and M = I - A^-1.  For a z-tension of +1%
    this returns eps_zz = +0.01 (matching recover_affine's eps_zz), whereas
    the affine's own per-axis scale A_zz = 1/(1-eps) ~= 1.0101 is the inverse
    (deformed-sampling) view of the same stretch.
    """
    return np.diag(np.eye(3) - np.linalg.inv(A))


def warp_onto_ref_grid(A, t, c, Nf, g_gpu, xp, chunk=30):
    """Warp g (GPU float32) onto f's voxel grid.

    For every voxel x in the reference grid, samples g at
        x' = A @ (x - c) + c + t
    with trilinear interpolation.  Out-of-bounds coordinates return 0
    (mode='constant'), so the valid region is exactly where the result > 0.

    Processing is done in chunks along axis-0 to limit peak GPU memory.

    Returns
    -------
    g_warped : float32 numpy array of shape Nf
    """
    from cupyx.scipy.ndimage import map_coordinates

    A_x = xp.asarray(A, dtype=xp.float32)
    t_x = xp.asarray(t, dtype=xp.float32).reshape(3, 1)
    c_x = xp.asarray(c, dtype=xp.float32).reshape(3, 1)

    g_warped = np.zeros(Nf, dtype=np.float32)
    n_chunks = (Nf[0] + chunk - 1) // chunk

    for idx, i0 in enumerate(range(0, Nf[0], chunk)):
        i1 = min(i0 + chunk, Nf[0])
        n  = i1 - i0

        ii = xp.arange(i0, i1, dtype=xp.float32)
        jj = xp.arange(0,  Nf[1], dtype=xp.float32)
        kk = xp.arange(0,  Nf[2], dtype=xp.float32)
        I, J, K = xp.meshgrid(ii, jj, kk, indexing='ij')
        pts = xp.stack([I.ravel(), J.ravel(), K.ravel()], axis=0)  # (3, N)
        del I, J, K, ii, jj, kk

        pts_g = A_x @ (pts - c_x) + c_x + t_x                     # (3, N)
        del pts

        # this is linear interpolation
        g_samp = map_coordinates(g_gpu, pts_g, order=1,
                                 mode='constant', cval=0.0)
        
        # this is cubic interpolation
        #g_samp = map_coordinates(g_gpu, pts_g, order=3,
        #                         mode='constant', cval=0.0, prefilter=True)
        del pts_g

        g_warped[i0:i1] = g_samp.reshape(n, Nf[1], Nf[2]).get()
        del g_samp

        if (idx + 1) % max(1, n_chunks // 5) == 0 or (idx + 1) == n_chunks:
            print(f"    {i1}/{Nf[0]} slices ({100*i1/Nf[0]:.0f}%)", flush=True)

    return g_warped


def bin_volume(vol, B):
    """Average-bin a 3D volume by integer factor B along every axis.

    vol.shape must be divisible by B in each dimension.
    Returns float32 array of shape (S0//B, S1//B, S2//B).

    Accumulates over the B**3 strided sub-slices directly instead of the
    reshape().mean() form.  The latter needs a full-resolution float32 copy
    of the input (12 bytes/voxel for a uint8 volume), which for a multi-
    gigavoxel CT volume exceeds host RAM and gets OOM-killed.  Here only the
    reduced-size accumulator (float32, 1/B**3 of the input) plus one strided
    slice are ever materialised, so peak memory stays a small fraction of the
    input.  For integer input the result is identical to reshape().mean()
    (float32 sums 0..255 exactly for any sane B).
    """
    S0, S1, S2 = vol.shape
    assert S0 % B == 0 and S1 % B == 0 and S2 % B == 0, \
        f"Volume shape {vol.shape} not divisible by bin factor {B}"
    out = np.zeros((S0 // B, S1 // B, S2 // B), dtype=np.float32)
    for i in range(B):
        for j in range(B):
            for k in range(B):
                out += vol[i::B, j::B, k::B]
    out /= B ** 3
    return out


def compute_crop(f, g_warped, h, glt=0.0):
    """Find n*h+1 crop of the valid overlap region (f>glt and g_warped>glt).

    Uses axis projections for an efficient bounding-box search, then snaps
    each dimension down to n*h+1 and re-centres.

    Parameters
    ----------
    h : int
        DVC element edge length; output size per axis = n*h+1.
    glt : float, default 0.0
        Gray-level threshold: a voxel counts as valid overlap only where both
        images exceed it (intensity <= glt is "no material / no contrast").

    Returns
    -------
    lo, hi : (3,) int arrays  (hi is exclusive)
    """
    valid = (f > glt) & (g_warped > glt)

    # bounding box via axis projections
    proj0 = valid.any(axis=(1, 2))
    proj1 = valid.any(axis=(0, 2))
    proj2 = valid.any(axis=(0, 1))

    idx0 = np.where(proj0)[0];  idx1 = np.where(proj1)[0];  idx2 = np.where(proj2)[0]
    if idx0.size == 0:
        raise RuntimeError("No valid overlap region found")

    lo   = np.array([idx0[0],  idx1[0],  idx2[0]])
    hi   = np.array([idx0[-1], idx1[-1], idx2[-1]])
    size = hi - lo + 1
    print(f"  valid overlap AABB : lo={lo}  hi={hi}  size={size}")

    # snap each dimension down to n*h+1
    snapped = ((size - 1) // h) * h + 1
    n_elems = (snapped - 1) // h
    print(f"  snapped size (h={h}): {snapped}  (n_elements = {n_elems})")

    # re-centre at snapped size
    mid    = (lo + hi) // 2
    lo_out = mid - snapped // 2
    hi_out = lo_out + snapped   # exclusive

    # shift into f's bounds if needed
    Nf = np.array(f.shape)
    for i in range(3):
        if lo_out[i] < 0:
            shift = -lo_out[i];  lo_out[i] += shift;  hi_out[i] += shift
        if hi_out[i] > Nf[i]:
            shift = hi_out[i] - Nf[i];  lo_out[i] -= shift;  hi_out[i] -= shift

    return lo_out, hi_out


# --------------------------------------------------------------------  main --

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ref_path', type=str, nargs='?',
                        help='Path to reference image (.npy file)')
    parser.add_argument('def_path', type=str, nargs='?',
                        help='Path to deformed image (.npy file)')
    parser.add_argument('--h', type=int, default=DEFAULT_H, dest='h',
                        help=f'DVC element edge length in voxels (default: {DEFAULT_H}). '
                             'Output image sizes satisfy size = n*h+1 per axis.')
    parser.add_argument('--glt', type=float, default=0.0, dest='glt',
                        help='Gray-level threshold marking voxels with no '
                             'information to correlate (default: 0.0). Voxels '
                             'with intensity <= glt (in both reference and '
                             'deformed images) are treated as no material / no '
                             'contrast and excluded from the overlap crop and '
                             'the correlation. Saved to affine_prealign.json and '
                             'reused by `voxeldvc run`.')
    parser.add_argument('--bin', type=int, default=1, dest='bin_factor',
                        help='Spatial binning factor (default: 1 = no binning). '
                             'Each output voxel is the mean of bin^3 input voxels. '
                             'The full-resolution crop is chosen as B*(n*h+1) so the '
                             'binned images are exactly n*h+1 per axis.')
    parser.add_argument('--decimate', type=int, default=4, dest='decimate',
                        help='Downsampling factor for affine pre-alignment (default: 4). '
                             'Larger is faster but coarser; 1 disables decimation.')
    parser.add_argument('--no-affine-refine', action='store_true',
                        dest='no_affine_refine',
                        help='Skip the full-resolution Gauss-Newton refinement of '
                             'the affine estimate. The coarse decimated ZNCC/Powell '
                             'prealign alone can miss sub-voxel strains (e.g. a few '
                             'percent compression); the refinement recovers them. '
                             'Only disable for debugging or if the refinement is '
                             'unstable on a particular dataset.')
    parser.add_argument('--output-dir', type=str, required=True, dest='output_dir',
                        help='Output directory (required).')
    parser.add_argument('--crop', type=int, nargs=3, default=[0, 0, 0], dest='crop',
                        metavar=('X', 'Y', 'Z'),
                        help='Pixels to crop symmetrically from each axis (default: 0 0 0). '
                             'E.g., --crop 100 0 0 removes 50 pixels from start and end of x-axis.')
    args = parser.parse_args(argv)

    # Validate required parameters
    if args.ref_path is None:
        print("ERROR: Missing required parameter 'ref_path'")
        print("Usage: python preprocess.py <ref_path> <def_path> [options]")
        print("       where <ref_path> is the path to the reference image (.npy file)")
        sys.exit(1)

    if args.def_path is None:
        print("ERROR: Missing required parameter 'def_path'")
        print("Usage: python preprocess.py <ref_path> <def_path> [options]")
        print("       where <def_path> is the path to the deformed image (.npy file)")
        sys.exit(1)

    h          = args.h
    glt        = args.glt
    bin_factor = args.bin_factor
    crop_vals  = args.crop
    decimate   = args.decimate
    ref_path   = os.path.expanduser(args.ref_path)
    def_path   = os.path.expanduser(args.def_path)

    if bin_factor < 1:
        print("ERROR: Invalid parameter '--bin': must be >= 1")
        print(f"       Received: {bin_factor}")
        sys.exit(1)

    if decimate < 1:
        print("ERROR: Invalid parameter '--decimate': must be >= 1")
        print(f"       Received: {decimate}")
        sys.exit(1)

    if any(c < 0 for c in crop_vals):
        print("ERROR: Invalid parameter '--crop': all values must be >= 0")
        print(f"       Received: {crop_vals}")
        sys.exit(1)

    # Validate input files exist
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference image file not found")
        print(f"       Parameter 'ref_path': {ref_path}")
        sys.exit(1)
    if not os.path.exists(def_path):
        print(f"ERROR: Deformed image file not found")
        print(f"       Parameter 'def_path': {def_path}")
        sys.exit(1)

    # Determine output directory (--output-dir is required)
    output_dir = os.path.expanduser(args.output_dir)

    # ---- Print supplied arguments ----
    print()
    print_params_box([
        ("ref_path",         ref_path),
        ("def_path",         def_path),
        ("h (element size)", h),
        ("glt (gray thresh)", glt),
        ("bin_factor",       bin_factor),
        ("crop (x,y,z)",     crop_vals),
        ("output_dir",       output_dir),
    ])
    print()

    print(f"Element edge length h = {h}  |  bin factor = {bin_factor}")
    if bin_factor > 1:
        print(f"  full-res crop sizes will be B*(n*h+1) = {bin_factor}*(n*{h}+1); "
              f"binned output sizes will be n*{h}+1")

    # ---- load ----
    print(f"Loading {ref_path}")
    ref = load_volume(ref_path)
    ref_original_dtype = ref.dtype
    print(f"  ref:      shape={ref.shape}  dtype={ref.dtype}")

    print(f"Loading {def_path}")
    deformed = load_volume(def_path)
    deformed_original_dtype = deformed.dtype
    print(f"  deformed: shape={deformed.shape}  dtype={deformed.dtype}")

    # ---- binning (first operation after load) ----
    if bin_factor > 1:
        # Trim each image to the largest shape divisible by bin_factor before binning
        ref_trim = tuple((s // bin_factor) * bin_factor for s in ref.shape)
        def_trim = tuple((s // bin_factor) * bin_factor for s in deformed.shape)
        # Pass the trimmed uint8 view straight to bin_volume; upcasting the
        # full-resolution volume to float32 here would need ~12 bytes/voxel
        # and OOM-kill on multi-gigavoxel CT data. bin_volume accumulates the
        # reduction without a full-resolution float32 copy.
        ref      = bin_volume(ref     [:ref_trim[0], :ref_trim[1], :ref_trim[2]], bin_factor)
        deformed = bin_volume(deformed[:def_trim[0], :def_trim[1], :def_trim[2]], bin_factor)
        print(f"\nAfter binning by {bin_factor}:")
        print(f"  ref:      shape={ref.shape}  dtype={ref.dtype}")
        print(f"  deformed: shape={deformed.shape}  dtype={deformed.dtype}")

    # ---- apply additional cropping (before affine pre-alignment) ----
    if any(cv > 0 for cv in crop_vals):
        crop_lo = [0, 0, 0]
        crop_hi = list(ref.shape)
        for i, cv in enumerate(crop_vals):
            if cv > 0:
                cv_start = cv // 2
                cv_end = cv - cv_start
                crop_lo[i] = cv_start
                crop_hi[i] -= cv_end
                if crop_lo[i] >= crop_hi[i]:
                    print(f"ERROR: crop value {cv} for axis {i} exceeds dimension {ref.shape[i]}")
                    sys.exit(1)
        ref      = ref     [crop_lo[0]:crop_hi[0], crop_lo[1]:crop_hi[1], crop_lo[2]:crop_hi[2]]
        deformed = deformed[crop_lo[0]:crop_hi[0], crop_lo[1]:crop_hi[1], crop_lo[2]:crop_hi[2]]
        print(f"\nAfter additional cropping:")
        print(f"  ref:      shape={ref.shape}  dtype={ref.dtype}")
        print(f"  deformed: shape={deformed.shape}  dtype={deformed.dtype}")

    # ---- affine pre-alignment ----
    xp    = cp
    f_pix = xp.asarray(ref.astype(np.float32))
    g_pix = xp.asarray(deformed.astype(np.float32))

    print(f"\nRunning affine_prealign (decimate={decimate}) …")
    A, t, c, info = affine_prealign(f_pix, g_pix, xp, decimate=decimate, disp=True)
    cp.cuda.Stream.null.synchronize()

    angles, scales = rotation_angles_deg(A)
    print_affine_box(A, angles, scales, t, c)
    print_cost_box(info)

    # ---- Gauss-Newton refinement (full resolution, coarse-to-fine) ----
    # The decimated ZNCC/Powell prealign above is robust for large rigid
    # offsets but coarse: a few-percent normal strain is sub-voxel once the
    # volume is decimated, so it can be missed entirely.  Refine (A, t) with
    # analytic Gauss-Newton at full resolution, which follows the sub-voxel
    # strain gradient directly (see affine_refine_gn).
    refine_info = None
    if not args.no_affine_refine:
        print("\nRefining affine by full-resolution Gauss-Newton "
              "(coarse-to-fine) …")
        A, t, refine_info = affine_refine_gn(f_pix, g_pix, xp, A, t, c,
                                             disp=False)
        cp.cuda.Stream.null.synchronize()
        angles, scales = rotation_angles_deg(A)
        print_affine_box(A, angles, scales, t, c)
        print_refine_box(refine_info)

    # f_pix no longer needed; keep g_pix for the warp
    del f_pix
    cp.get_default_memory_pool().free_all_blocks()

    # ---- warp deformed onto reference grid ----
    print("\nWarping deformed onto reference grid …")
    g_warped = warp_onto_ref_grid(A, t, c, ref.shape, g_pix, xp, chunk=30)

    del g_pix
    cp.get_default_memory_pool().free_all_blocks()

    nz_pct = 100.0 * float(np.count_nonzero(g_warped > glt)) / g_warped.size
    print(f"  g_warped: shape={g_warped.shape}  above-glt={nz_pct:.1f}%")

    # ---- find overlap crop ----
    print("\nFinding overlap crop …")
    lo, hi = compute_crop(ref, g_warped, h, glt)
    del g_warped  # used only to find the crop; not saved

    crop_shape = tuple(int(x) for x in (hi - lo))
    print(f"\nFinal crop shape : {crop_shape}")
    print(f"  slice  [{lo[0]}:{hi[0]}, {lo[1]}:{hi[1]}, {lo[2]}:{hi[2]}]")
    print(f"  coverage : ref {100*np.prod(crop_shape)/ref.size:.1f}%")

    for i, s in enumerate(crop_shape):
        n = (s - 1) // h
        assert s == n * h + 1, f"axis {i}: {s} is not n*{h}+1"
        print(f"  axis {i} : {s} = {n}*{h}+1  ✓")

    ref_crop        = ref     [lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    def_ref_aligned = deformed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # ---- compute deformed-space bounding box for correlation ----
    # Map the 8 corners of the reference crop through the affine to find the
    # range of deformed-image coordinates sampled during correlation.  Add a
    # 1-voxel margin on each side for trilinear interpolation safety, then
    # clamp to the deformed image extent.
    corners_ref = np.array([
        [lo[0] + dx * (hi[0] - lo[0] - 1),
         lo[1] + dy * (hi[1] - lo[1] - 1),
         lo[2] + dz * (hi[2] - lo[2] - 1)]
        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
    ], dtype=np.float64)  # (8, 3)  full-image reference coords

    corners_def = (corners_ref - c) @ A.T + c + t   # (8, 3)  deformed coords

    lo_def = np.floor(corners_def.min(axis=0)).astype(int) - 1
    hi_def = np.ceil( corners_def.max(axis=0)).astype(int) + 2  # exclusive
    lo_def = np.maximum(lo_def, 0)
    hi_def = np.minimum(hi_def, np.array(deformed.shape))

    def_crop = deformed[lo_def[0]:hi_def[0], lo_def[1]:hi_def[1], lo_def[2]:hi_def[2]]
    print(f"\nDeformed bounding box for correlation:")
    print(f"  lo_def = {lo_def.tolist()}  hi_def = {hi_def.tolist()}")
    print(f"  def_preprocessed shape = {def_crop.shape}")

    # Convert crops back to original dtypes
    ref_crop        = ref_crop.astype(ref_original_dtype)
    def_crop        = def_crop.astype(deformed_original_dtype)
    def_ref_aligned = def_ref_aligned.astype(deformed_original_dtype)

    out_ref         = os.path.join(output_dir, 'ref_preprocessed.npy')
    out_def         = os.path.join(output_dir, 'def_preprocessed.npy')
    out_ref_aligned = os.path.join(output_dir, 'def_ref_aligned.npy')
    out_json        = os.path.join(output_dir, 'affine_prealign.json')

    os.makedirs(output_dir, exist_ok=True)

    np.save(out_ref, ref_crop)
    np.save(out_def, def_crop)
    np.save(out_ref_aligned, def_ref_aligned)
    print(f"\nSaved {out_ref}         shape={ref_crop.shape}  dtype={ref_crop.dtype}")
    print(f"Saved {out_def}         shape={def_crop.shape}  dtype={def_crop.dtype}")
    print(f"Saved {out_ref_aligned}  shape={def_ref_aligned.shape}  dtype={def_ref_aligned.dtype}")

    # ---- write JSON with affine parameters and crop metadata ----
    # All coordinates (A, t, c, crop_lo) are in the binned voxel space that
    # the DVC solver operates in.  To reconstruct full-resolution quantities:
    #   t_fullres = t * bin_factor,  c_fullres = c * bin_factor  (A unchanged)
    # For a DVC node at output position x_out (binned voxel index):
    #   u_affine_binned = (A - I) @ (x_out + crop_lo - c) + t
    #   u_total_fullres = (u_affine_binned + u_dvc) * bin_factor
    angles, scales = rotation_angles_deg(A)
    meta = {
        "description": (
            "Affine pre-alignment parameters. "
            "glt is the gray-level threshold below which a voxel is treated as "
            "no material / no contrast (excluded from the overlap crop and the "
            "correlation); it is reused by run_dvc.py. "
            "All spatial quantities (t, c, crop_lo, def_lo) are in binned voxel units. "
            "A is dimensionless (rotation+scale) and the same in both spaces. "
            "def_preprocessed.npy is the ORIGINAL (non-warped) deformed volume "
            "cropped to the affine-mapped bounding box [def_lo, def_hi); used by "
            "run_dvc.py with ext_affine to avoid double interpolation. "
            "def_ref_aligned.npy is the deformed volume at the same index range as "
            "the reference crop [crop_lo, crop_hi); used by run_dvc.py for "
            "write_outputs (saved as deformed.npy) without reloading the original file. "
            "For a DVC node at output position x_out: "
            "u_affine = (A-I) @ (x_out + crop_lo - c) + t  [binned voxels]; "
            "u_total_fullres = (u_affine + u_dvc) * bin_factor."
        ),
        "h": h,
        "glt": glt,
        "bin_factor": bin_factor,
        "additional_crop_xyz": crop_vals,
        "ref_file":  ref_path,
        "def_file":  def_path,
        "ref_crop_file":     out_ref,
        "def_crop_file":     out_def,
        "def_ref_aligned_file": out_ref_aligned,
        "crop_lo":    [int(x) for x in lo],
        "crop_hi":    [int(x) for x in hi],
        "crop_shape": list(crop_shape),
        "def_lo":     [int(x) for x in lo_def],
        "def_hi":     [int(x) for x in hi_def],
        "affine_A":   A.tolist(),
        "affine_t":   t.tolist(),
        "affine_c":   c.tolist(),
        "affine_t_fullres": (t * bin_factor).tolist(),
        "affine_c_fullres": (c * bin_factor).tolist(),
        "rotation_deg_xyz": [float(angles[0]), float(angles[1]), float(angles[2])],
        "scale_xyz":        [float(scales[0]), float(scales[1]), float(scales[2])],
        "normal_strain_xyz": [float(e) for e in normal_strains(A)],
        "cost_identity":         float(info['cost_identity']),
        "cost_translation_only": float(info['cost_translation_only']),
        "cost_affine":           float(info['cost_final']),
        "nfev": int(info['nfev']),
        "affine_refine": (None if refine_info is None else {
            "rms_initial": float(refine_info['rms_initial']),
            "rms_final":   float(refine_info['rms_final']),
            "iters":       [int(n) for n in refine_info['iters']],
        }),
    }
    with open(out_json, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved {out_json}")


if __name__ == '__main__':
    main()
