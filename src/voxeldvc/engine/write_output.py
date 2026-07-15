# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-15 20:43:07
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-15 21:28:32

"""Shared output-generation helpers for the run_*.py DVC driver scripts.

`write_outputs(...)` takes the raw correlation result (`U`, `res`) plus the
reference/deformed volumes and mesh parameters, computes all derived fields
(reshaped displacement/residual grids, cell-centered principal strains and
their deformed-configuration remap, the unsafe voxel mask and its
forward-push, and the overlap-lost element mask and its deformed-config
remap), prints summary statistics matching the existing run scripts, and
saves all `.npy` outputs to `OUTPUT_DIR`.

Outputs written by `write_outputs`:
  - U_recovered.npy   : (Nx+1, Ny+1, Nz+1, 3) nodal displacement field
                        (node-major DOFs, order='N', x fastest -> 'F'
                        reshape).
  - residual.npy      : (Nvx, Nvy, Nvz) per-voxel residual field.
  - ref.npy           : (Nvx, Nvy, Nvz) reference volume (the region the
                        correlation was run on / outputs are aligned to).
  - deformed.npy      : (Nvx, Nvy, Nvz) deformed volume, same region as
                        ref.npy.
  - principal_strains_cell.npy : (Nx, Ny, Nz, 3) principal small-strain
                        values e1>=e2>=e3 at the center of each element,
                        from the eigenvalues of the symmetric small-strain
                        tensor eps_ij = 0.5*(du_i/dx_j + du_j/dx_i), with
                        gradients evaluated at the element centroid via the
                        Hex8 shape-function gradients (see
                        compute_principal_strains_cell) applied to the 8
                        corner nodes' displacements.
  - principal_strains_cell_deformed.npy : (Nx, Ny, Nz, 3)
                        principal_strains_cell remapped onto the deformed
                        configuration: each reference element center is
                        moved by the average nodal displacement of its 8
                        corners, and the field values at these scattered
                        deformed positions are interpolated (griddata,
                        linear, with nearest-neighbour fill outside the
                        convex hull) back onto the regular element-center
                        grid -- so this field is spatially aligned with
                        deformed.npy.
  - equivalent_strain_cell.npy : (Nx, Ny, Nz) von Mises equivalent strain
                        eps_eq = (sqrt(2)/3) *
                        sqrt((e1-e2)^2+(e2-e3)^2+(e3-e1)^2) at each element
                        center, derived from principal_strains_cell -- a
                        single non-negative deviatoric-magnitude scalar per
                        element (see compute_equivalent_strain).
  - equivalent_strain_cell_deformed.npy : (Nx, Ny, Nz) the equivalent strain
                        of principal_strains_cell_deformed, i.e. the same
                        von Mises scalar computed on the deformed-configuration
                        principal strains, so it is spatially aligned with
                        deformed.npy.
  - residual_std_cell.npy : (Nx, Ny, Nz) per-element population std of the
                        final residual over the element's active (res != 0)
                        voxels, in grey-level units -- the per-cell analog of
                        the global `std(res)` correlate_gpu prints each GN
                        iteration (a "did the correlation succeed here" quality
                        field). NaN where an element has too few active voxels.
                        See compute_correlation_quality_cell.
  - zncc_cell.npy     : (Nx, Ny, Nz) per-element zero-normalized cross-
                        correlation between ref and the warped deformed image
                        (ref - res) over the element's active voxels, in
                        [-1, 1] (1 = perfect local fit). Intensity-scale-
                        independent per-cell quality; reliably flags local
                        divergence. NaN where too few active voxels / degenerate
                        variance. See compute_correlation_quality_cell.
  - gn_convergence_cell.npy : (Nx, Ny, Nz) per-element Gauss-Newton convergence
                        ratio ||dU||/||U|| over the element's 8 corner nodes --
                        the local analog of the global dU/U the solve stops on
                        (small = settled, large = the last step still moved the
                        element). Only written when the final update `dU` is
                        passed to write_outputs. NaN where ||U||~0. See
                        compute_gn_convergence_cell.
  - unsafe_voxel_mask.npy : (Nvx, Nvy, Nvz) bool, per-voxel mask of voxels
                        affected by an "unsafe" node -- a mesh node that
                        belongs to an element containing at least one voxel
                        outside the final dynamic mask (lost ref/deformed
                        overlap). Flags the one-element dilation of the
                        lost-overlap region, where the correlation result
                        may be dominated by regularization rather than
                        image data. The outer one-element boundary shell is
                        always flagged too (regularizer Neumann-BC strain
                        bias, see boundary_element_mask / TUTORIAL 5.4).
  - overlap_lost_element_mask.npy : (Nx, Ny, Nz) bool, cell-centered (same
                        grid as principal_strains_cell.npy). The mask to gate
                        the cell-strain output: the strain is a trustworthy
                        measurement on its complement (False entries). An
                        element is flagged True if EITHER (1) its *material*
                        (mask0=True) lost correlation overlap (deformed
                        position out of bounds or onto g<=0; 1-element dilated
                        through shared nodes), OR (2) its material count
                        fraction -- (# mask0=True voxels)/(h+1)^3 -- is below
                        0.5 (compute_overlap_lost_element_mask's
                        min_material_frac default), i.e. the element is mostly
                        pore/air so its strain is interpolation/regularization-
                        governed rather than measured. Unlike the
                        porosity-conflating unsafe_voxel_mask, well-correlated
                        porosity that does not lose overlap and meets the
                        material floor is NOT flagged. (Cell strain is a
                        function of the nodal displacements only, so porosity
                        per se does not invalidate it -- criterion 2 is the
                        "is there enough material to measure" test, not a
                        validity test.) A third criterion always flags the
                        outer one-element boundary shell (regularizer Neumann-BC
                        strain bias, see boundary_element_mask / TUTORIAL 5.4).
                        See compute_overlap_lost_element_mask in
                        src/kernels_dvc.py for the full definition.
  - overlap_lost_element_mask_deformed.npy : (Nx, Ny, Nz) bool,
                        overlap_lost_element_mask remapped onto the deformed
                        configuration (same nearest-neighbour element-center
                        remap as principal_strains_cell_deformed: each
                        reference element center is moved by the average nodal
                        displacement of its 8 corners), so it gates
                        principal_strains_cell_deformed.npy.
  - ref_forward_pushed.npy : (Nvx, Nvy, Nvz) reference volume pushed forward
                        by the recovered displacement field onto the
                        deformed-image grid, i.e.
                        ref_forward_pushed(x) = ref(x - u(x)) with u(x) the
                        FE-interpolated displacement at voxel x (same
                        interpolation as compute_active_mask /
                        interp_field_vox), sampled via trilinear
                        map_coordinates (mode='nearest') -- the
                        small-deformation approximation g(x) ~= f(x-u(x))
                        for a converged correlation. This is the predicted
                        deformed image and is directly comparable to
                        deformed.npy.
  - dvc_fields.vti : binary XML VTK ImageData (ParaView/VisIt-ready) holding
                        the nodal displacement field as POINT data
                        (`displacement`, 3-component, from U_recovered) and, as
                        CELL data, the principal strains eps1>=eps2>=eps3, the
                        equivalent strain eps_eq, the material_mask, and the
                        per-cell convergence/quality fields residual_std, zncc
                        (and gn_convergence when `dU` is supplied), on one
                        (Nx+1,Ny+1,Nz+1)-point /
                        (Nx,Ny,Nz)-cell grid with spacing=(h,h,h) (node
                        positions in voxel units, so the voxel-unit displacement
                        warps correctly). Written via the pure-NumPy
                        save_dvc_fields_vti in voxeldvc.vtk_export
                        (see _load_vtk_export); the export is
                        guarded so a missing/broken vtk_export never fails a run.
  - unsafe_voxel_mask_forward_pushed.npy : (Nvx, Nvy, Nvz) bool, the
                        reference-configuration unsafe_voxel_mask.npy pushed
                        forward onto the deformed-image grid the same way as
                        ref_forward_pushed.npy (see push_field_forward), but
                        with nearest-neighbour (order=0) sampling and
                        mode='constant', cval=1 (rather than 'nearest')
                        since the field is boolean and out-of-domain source
                        coordinates must be flagged unsafe rather than
                        inheriting a clamped edge value -- otherwise a large
                        rigid shift causes one boundary "unsafe shell" to be
                        lost off the domain while the opposite edge's value
                        is merely duplicated, shrinking the apparent unsafe
                        fraction without any real change in correlation
                        overlap. Spatially aligned with
                        deformed.npy / ref_forward_pushed.npy.

Field types (mesh grid the output is sampled on):
  - Vertex (nodal) data, shape (Nx+1, Ny+1, Nz+1[, ...]), one value per mesh
    node:
      U_recovered.npy.
  - Cell (element-centered) data, shape (Nx, Ny, Nz[, ...]), one value per
    mesh element:
      principal_strains_cell.npy, principal_strains_cell_deformed.npy,
      equivalent_strain_cell.npy, equivalent_strain_cell_deformed.npy,
      overlap_lost_element_mask.npy, overlap_lost_element_mask_deformed.npy,
      residual_std_cell.npy, zncc_cell.npy, gn_convergence_cell.npy.
  - Voxel data, shape (Nvx, Nvy, Nvz) = (Nx*h+1, Ny*h+1, Nz*h+1), one value
    per CT voxel:
      residual.npy, ref.npy, deformed.npy,
      unsafe_voxel_mask.npy, ref_forward_pushed.npy,
      unsafe_voxel_mask_forward_pushed.npy.
"""

import os

import numpy as np
from scipy.interpolate import griddata

from .correlate_gpu import compute_active_mask, interp_field_vox, _map_coordinates
from .geometry_dvc import build_N_stencil, _shape_grads
from .kernels_dvc import (compute_unsafe_voxel_mask,
                                  compute_overlap_lost_element_mask)


def _load_vtk_export():
    """Import the ``voxeldvc.vtk_export`` module (the .vti/.vtk writers).

    Kept as a lazily-called helper so the .vti export stays guarded: a broken
    or missing writer must never fail an otherwise-complete correlation run.
    Returns the loaded module, or ``None`` if it cannot be imported.
    """
    try:
        from .. import vtk_export
    except Exception:
        return None
    return vtk_export


def _resolve_xp(xp):
    """Return the array module to run the voxel-scale output stages on.

    The heavy derived-field work here (compute_active_mask, the forward
    pushes, ZNCC) is dominated by `interp_field_vox` + `map_coordinates` over
    full `Nvox`-sized arrays, which is ~50x faster on the GPU (see the
    element-chunked GPU kernels in correlate_gpu). `xp=None` auto-selects
    cupy when it is importable and a device is present, else numpy, so the
    output stage inherits the GPU the correlation just ran on without the
    caller having to plumb it through. Pass an explicit module to force it.
    """
    if xp is not None:
        return xp
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except Exception:
        pass
    return np


def _to_numpy(a, xp):
    """Bring an xp array back to host numpy for np.save / CPU-only stages."""
    return a if xp is np else xp.asnumpy(a)


def mesh_dims(shape, h):
    """Number of elements per axis: largest N such that N*h <= shape-1."""
    return tuple((s - 1) // h for s in shape)


class Tee:
    """Writes to multiple streams (e.g. stdout and a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# Local node ordering for the 8 corners of a Hex8 element, matching
# geometry_dvc.py's element_node_ids / build_K_ref:
#   0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
#   4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
_HEX8_CORNER_OFFSETS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def compute_principal_strains_cell(U_grid, h):
    """Principal small-strain values e1>=e2>=e3 at the center of every
    element of U_grid (shape (Nx+1,Ny+1,Nz+1,3), last axis = (ux,uy,uz)),
    via the Hex8 FE shape functions.

    For each element, grad_u[..., i, j] = d(u_i)/d(x_j) is evaluated at
    the element centroid (natural coordinates (0.5,0.5,0.5)) from the 8
    corner nodes' displacements using the Hex8 shape-function gradients
    (geometry_dvc._shape_grads), scaled by 1/h for the physical (voxel)
    element size h. eps = 0.5*(grad_u + grad_u.T); principal strains are
    the eigenvalues of eps, sorted descending. Returns shape (Nx,Ny,Nz,3).
    """
    dN = _shape_grads(0.5, 0.5, 0.5) / h  # (3, 8): dN_a/dx_d at centroid

    Nx, Ny, Nz = (n - 1 for n in U_grid.shape[:3])
    U_corners = np.stack([
        U_grid[di:di + Nx, dj:dj + Ny, dk:dk + Nz, :]
        for di, dj, dk in _HEX8_CORNER_OFFSETS
    ], axis=-2)  # (Nx,Ny,Nz,8,3)

    # grad_u[...,c,d] = d(u_c)/d(x_d) = sum_a U_corners[...,a,c] * dN[d,a]
    grad_u = np.einsum('...ac,da->...cd', U_corners, dN)

    eps = 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))
    eigvals = np.linalg.eigvalsh(eps)  # ascending order
    return eigvals[..., ::-1]  # descending: e1>=e2>=e3


def compute_equivalent_strain(principal_strains):
    """Von Mises equivalent strain from the principal small-strain values.

    `principal_strains` has shape (..., 3) with the three principal strains
    (e1, e2, e3) along the last axis (order is irrelevant -- the result is
    symmetric in the three values). Returns an array of shape
    principal_strains.shape[:-1].

    eps_eq = sqrt(2/3 * eps_dev : eps_dev), which for principal strains is

        eps_eq = (sqrt(2)/3) *
                 sqrt((e1-e2)^2 + (e2-e3)^2 + (e3-e1)^2)

    a single non-negative scalar per element summarizing the deviatoric
    (distortional) strain magnitude, independent of the hydrostatic part.
    This is the standard equivalent-strain scalar used to visualize a strain
    field with one channel instead of three principal components.
    """
    e1 = principal_strains[..., 0]
    e2 = principal_strains[..., 1]
    e3 = principal_strains[..., 2]
    return (np.sqrt(2.0) / 3.0) * np.sqrt(
        (e1 - e2) ** 2 + (e2 - e3) ** 2 + (e3 - e1) ** 2)


def compute_material_cell_mask(ref, h, Nx, Ny, Nz, glt=0.0):
    """Per-element bool mask, True where the *mean* reference gray level over
    the element's (h+1)^3 voxels exceeds the gray-level threshold `glt`.

    Uses the same element->voxel mapping as the porosity/overlap kernels
    (kernels_dvc.get_voxel_offsets / get_elem_voxel_origins): element (i,j,k)
    owns the (h+1)^3 block of reference voxels at h*(i,j,k) .. h*(i,j,k)+h
    (shared faces included), so it is consistent with
    compute_overlap_lost_element_mask's material-fraction test -- but this mask
    thresholds the block *mean intensity* rather than a per-voxel material
    count. Returns shape (Nx,Ny,Nz), indexed [i,j,k]=[x,y,z] (same cell grid
    as principal_strains_cell), so it gates the .vti cell fields.
    """
    from .kernels_dvc import get_voxel_offsets, get_elem_voxel_origins

    ref_flat = np.asarray(ref).reshape(-1, order='F')      # x-fastest voxels
    elem_ids = np.arange(Nx * Ny * Nz)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, np)
    elem_origins = get_elem_voxel_origins(elem_ids, h, Nx, Ny, np)
    elem_voxels = elem_origins[:, None] + voxel_offsets[None, :]  # (Nelem,(h+1)^3)

    mean_gray = ref_flat[elem_voxels].mean(axis=1)         # x-fastest elem order
    # elem_id = i + j*Nx + k*Nx*Ny (x fastest) -> (Nx,Ny,Nz) via order='F',
    # matching how overlap_lost_element_mask is reshaped in write_outputs.
    return (mean_gray > glt).reshape(Nx, Ny, Nz, order='F')


def compute_material_cell_mask_eroded(material_cell_mask):
    """One-cell erosion of `material_cell_mask`: True only where the element is
    itself material *and* none of its 26 neighbours (face/edge/corner) is a
    non-material (False) element. Any element touching an already-False element
    is set to False, shrinking the material region by one element layer.

    The domain boundary is *not* treated as False (border_value=1), so elements
    are eroded only by genuine internal False neighbours, not by the mesh edge.
    Returns a bool array with the same (Nx,Ny,Nz) shape/indexing as the input.
    """
    from scipy.ndimage import binary_erosion

    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity ("touching")
    return binary_erosion(material_cell_mask, structure=structure, border_value=1)


def compute_safe_cell_mask(unsafe_vox, h, Nx, Ny, Nz):
    """Per-element "safe" mask reduced from the per-voxel unsafe_voxel_mask.

    True where the element is *safe* (inverse-of-unsafe convention, matching
    view_reference.py's `~unsafe_voxel_mask` overlay): an element is safe only
    if none of its (h+1)^3 voxels is flagged unsafe. Since unsafe_voxel_mask is
    already a one-element dilation of the lost-overlap region, this is the
    conservative cell-grid gate -- any element touching an unsafe voxel is
    dropped.

    `unsafe_vox` is the flat (x-fastest) unsafe voxel mask, before its reshape
    to the (Nvx,Nvy,Nvz) grid. Uses the same element->voxel mapping as
    compute_material_cell_mask, and returns shape (Nx,Ny,Nz) indexed
    [i,j,k]=[x,y,z] -- the same cell grid as principal_strains_cell -- so it
    gates the .vti cell fields.
    """
    from .kernels_dvc import get_voxel_offsets, get_elem_voxel_origins

    unsafe_flat = np.asarray(unsafe_vox).reshape(-1, order='F')  # x-fastest voxels
    elem_ids = np.arange(Nx * Ny * Nz)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, np)
    elem_origins = get_elem_voxel_origins(elem_ids, h, Nx, Ny, np)
    elem_voxels = elem_origins[:, None] + voxel_offsets[None, :]  # (Nelem,(h+1)^3)

    any_unsafe = unsafe_flat[elem_voxels].any(axis=1)             # x-fastest elem order
    return (~any_unsafe).reshape(Nx, Ny, Nz, order='F')


def compute_correlation_quality_cell(res, ref, h, Nx, Ny, Nz):
    """Per-element correlation-quality fields derived from the final per-voxel
    residual `res` (option 1: "did the correlation succeed here").

    For each element the reduction runs over its (h+1)^3 quadrature voxels
    (the same element->voxel mapping as compute_material_cell_mask /
    kernels_dvc.get_voxel_offsets), restricted to that element's *active*
    voxels -- those with res != 0, which is exactly the b_mask the solve
    summed over (correlate_gpu zeroes res outside b_mask, and write_outputs
    already uses res != 0 as the active-voxel test).

    Returns (residual_std_cell, zncc_cell), both (Nx,Ny,Nz), indexed
    [i,j,k]=[x,y,z] (same cell grid as principal_strains_cell):
      residual_std_cell : population std of res over the element's active
          voxels, in grey-level units -- the per-cell analog of the
          `std(res)=.. gl` figure correlate_gpu prints globally each GN
          iteration. Lower = the recovered displacement fits the images
          better in that element.
      zncc_cell : zero-normalized cross-correlation between the reference and
          the warped deformed image (g_warped = ref - res) over the element's
          active voxels, in [-1, 1] (1 = perfect local correlation). The
          per-cell analog of write_outputs' global ZNCC proxy (see
          compute_zncc): intensity-scale-independent and comparable across
          cells, and (per CLAUDE.md) a reliable *divergence* flag.
    Elements with fewer than `min_active` (=8) active voxels or degenerate
    variance are NaN in both fields (too little material to score).
    """
    from .kernels_dvc import get_voxel_offsets, get_elem_voxel_origins

    min_active = 8
    res_flat = np.asarray(res).reshape(-1, order='F').astype(np.float64)
    ref_flat = np.asarray(ref).reshape(-1, order='F').astype(np.float64)

    Ne = Nx * Ny * Nz
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, np)
    elem_origins = get_elem_voxel_origins(np.arange(Ne), h, Nx, Ny, np)
    vox = elem_origins[:, None] + voxel_offsets[None, :]     # (Ne, (h+1)^3)

    r = res_flat[vox]                                        # (Ne, nq)
    f = ref_flat[vox]
    g = f - r                                                # warped deformed
    m = (r != 0.0).astype(np.float64)                        # active voxels
    cnt = m.sum(axis=1)
    denom_cnt = np.maximum(cnt, 1.0)

    # Per-cell residual std (population, mean removed per element).
    mean_r = (r * m).sum(axis=1) / denom_cnt
    var_r = (r * r * m).sum(axis=1) / denom_cnt - mean_r ** 2
    residual_std = np.sqrt(np.clip(var_r, 0.0, None))

    # Per-cell ZNCC between f and g over the active voxels.
    fm = (f * m).sum(axis=1) / denom_cnt
    gm = (g * m).sum(axis=1) / denom_cnt
    fc = (f - fm[:, None]) * m
    gc = (g - gm[:, None]) * m
    num = (fc * gc).sum(axis=1)
    den = np.sqrt((fc * fc).sum(axis=1) * (gc * gc).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        zncc = num / den

    bad = (cnt < min_active) | (den <= 0.0)
    residual_std[cnt < min_active] = np.nan
    zncc[bad] = np.nan

    # elem_id = i + j*Nx + k*Nx*Ny (x fastest) -> (Nx,Ny,Nz) via order='F',
    # matching overlap_lost_element_mask / material_cell_mask.
    return (residual_std.reshape(Nx, Ny, Nz, order='F'),
            zncc.reshape(Nx, Ny, Nz, order='F'))


def compute_gn_convergence_cell(dU, U, Nx, Ny, Nz):
    """Per-element Gauss-Newton convergence ratio ||dU||/||U|| (option 2: "how
    converged is the iterate here"), the local analog of the global dU/U that
    correlate_gpu prints and stops on.

    `dU` is the final GN update returned by correlate_gpu and `U` the converged
    displacement, both flat (Ndof,) node-major DOF vectors (node = i +
    j*(Nx+1) + k*(Nx+1)*(Ny+1), x fastest; component fastest within a node).
    For each element the ratio is the Euclidean norm of dU over its 8 corner
    nodes' 24 DOFs divided by the same norm of U -- small where the iterate has
    settled, large where the last step still moved the element appreciably
    (under-converged / drifting).

    Returns (Nx,Ny,Nz), indexed [i,j,k]=[x,y,z] (same cell grid as
    principal_strains_cell). Elements whose ||U|| is ~0 are NaN (ratio
    undefined).
    """
    nbx, nby, nbz = Nx + 1, Ny + 1, Nz + 1

    def to_grid3(vec):
        # (Ndof,) -> (nbx,nby,nbz,3), last axis (x,y,z) DOF component.
        return np.stack(
            [vec[c::3].reshape(nbx, nby, nbz, order='F') for c in range(3)],
            axis=-1)

    dU_grid = to_grid3(np.asarray(dU, dtype=np.float64))
    U_grid = to_grid3(np.asarray(U, dtype=np.float64))

    def corner_sqnorm(grid):
        # sum over the 8 corner nodes and 3 components of grid**2 -> (Nx,Ny,Nz).
        acc = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        for di, dj, dk in _HEX8_CORNER_OFFSETS:
            block = grid[di:di + Nx, dj:dj + Ny, dk:dk + Nz, :]
            acc += (block * block).sum(axis=-1)
        return acc

    num = np.sqrt(corner_sqnorm(dU_grid))
    den = np.sqrt(corner_sqnorm(U_grid))
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = num / den
    ratio[den <= 0.0] = np.nan
    return ratio


def remap_element_mask_to_deformed(elem_mask, U_grid, h, crop_lo=(0, 0, 0)):
    """Remap a per-element bool mask (shape (Nx,Ny,Nz)) from the reference
    configuration onto the deformed configuration.

    Each reference element center ((i+0.5)*h,(j+0.5)*h,(k+0.5)*h) is moved
    to its deformed position by the average of its 8 corner nodes'
    displacements (from U_grid); `elem_mask` values at these scattered
    deformed positions are remapped (nearest-neighbour) back onto the
    regular element-center grid, so the result is spatially aligned with
    the deformed image.

    crop_lo : (3,) int-like -- offset of the reference crop origin in the
        full-image coordinate frame.  U_grid encodes displacements to
        full-image deformed coordinates, so scatter positions x_def are also
        in full-image deformed coords.  The query positions must be in the
        same frame: full-image deformed position of output element (i,j,k) =
        (i+0.5)*h + crop_lo.  For callers without a pre-alignment crop
        (crop_lo = (0,0,0)) this is a no-op.
    """
    Nx, Ny, Nz = elem_mask.shape
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    x_ref = (np.stack([ii, jj, kk], axis=-1).astype(np.float64) + 0.5) * h

    u_avg = np.zeros(x_ref.shape, dtype=np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                u_avg += U_grid[ii + di, jj + dj, kk + dk]
    u_avg /= 8.0
    x_def = x_ref + u_avg  # full-image deformed coords

    points = x_def.reshape(-1, 3)
    query = (x_ref + np.asarray(crop_lo, dtype=np.float64)).reshape(-1, 3)
    values = elem_mask.reshape(-1).astype(np.float64)

    out = griddata(points, values, query, method='nearest')
    return out.reshape(elem_mask.shape).astype(bool)


def remap_cell_field_to_deformed(field, U_grid, h, crop_lo=(0, 0, 0)):
    """Remap a per-element field (shape (Nx,Ny,Nz,...)) from the reference
    configuration onto the deformed configuration.

    Each reference element center ((i+0.5)*h,(j+0.5)*h,(k+0.5)*h) is moved
    to its deformed position by the average of its 8 corner nodes'
    displacements (from U_grid); `field` values at these scattered
    deformed positions are interpolated (linear, nearest-neighbour fill
    outside the convex hull) back onto the regular element-center grid, so
    the result is spatially aligned with the deformed image. The cell-field
    (interpolated) analogue of remap_element_mask_to_deformed (nearest).

    crop_lo : (3,) int-like -- see remap_element_mask_to_deformed.
    """
    Nx, Ny, Nz = field.shape[:3]
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    x_ref = (np.stack([ii, jj, kk], axis=-1).astype(np.float64) + 0.5) * h

    u_avg = np.zeros(x_ref.shape, dtype=np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                u_avg += U_grid[ii + di, jj + dj, kk + dk]
    u_avg /= 8.0
    x_def = x_ref + u_avg  # full-image deformed coords

    points = x_def.reshape(-1, 3)
    query = (x_ref + np.asarray(crop_lo, dtype=np.float64)).reshape(-1, 3)
    values = field.reshape(-1, field.shape[-1]) if field.ndim == 4 else field.reshape(-1, 1)

    out_lin = griddata(points, values, query, method='linear')
    out_nearest = griddata(points, values, query, method='nearest')
    nan_mask = np.isnan(out_lin)
    out_lin[nan_mask] = out_nearest[nan_mask]

    return out_lin.reshape(field.shape)


def push_field_forward(field, U, Nx_e, Ny_e, Nz_e, h, order=1, mode='nearest',
                       cval=0.0, xp=np):
    """Push a per-voxel field forward by the recovered displacement field
    onto the deformed-image grid: returns field_pushed with
    field_pushed(x) = field(x - u(x)), where u(x) is the FE-interpolated
    displacement at voxel x (same interp_field_vox used by
    compute_active_mask), sampled via map_coordinates (mode=`mode`,
    cval=`cval`, order=`order`). `field` must have shape (Nvx,Nvy,Nvz)
    matching the voxel grid implied by Nx_e,Ny_e,Nz_e,h. Use order=1
    (linear) for intensity fields (see push_reference_forward) and order=0
    (nearest) for label/mask fields, to avoid interpolated in-between
    values. `mode='nearest'` (the default) clamps out-of-domain lookups to
    the nearest edge voxel, which for a field with a "shell" of flagged
    voxels at the domain boundary (e.g. unsafe_voxel_mask) causes that
    shell to be lost off one side of the domain under a large rigid shift
    while the opposite edge value gets duplicated -- use
    `mode='constant', cval=<flag value>` instead so any voxel whose source
    falls outside the reference domain is flagged rather than silently
    inheriting a clamped edge value.

    xp : numpy or cupy module -- interp_field_vox and map_coordinates run on
        this backend; the returned array matches it. All the work is over
        full Nvox-sized arrays, so this is the largest CPU->GPU win in the
        output stage (see write_outputs).
    """
    Nvx, Nvy, Nvz = field.shape
    N_stencil, _ = build_N_stencil(h)
    u_vox = interp_field_vox(xp.asarray(U), N_stencil, h, Nx_e, Ny_e, Nz_e, xp)

    vx_idx, vy_idx, vz_idx = xp.meshgrid(
        xp.arange(Nvx), xp.arange(Nvy), xp.arange(Nvz), indexing='ij')
    vx_idx = xp.ravel(vx_idx, order='F').astype(u_vox.dtype)
    vy_idx = xp.ravel(vy_idx, order='F').astype(u_vox.dtype)
    vz_idx = xp.ravel(vz_idx, order='F').astype(u_vox.dtype)

    coords = xp.stack([vx_idx - u_vox[:, 0], vy_idx - u_vox[:, 1], vz_idx - u_vox[:, 2]], axis=0)
    pushed = _map_coordinates(xp.asarray(field), coords, xp, order=order, mode=mode, cval=cval)
    return pushed.reshape(Nvx, Nvy, Nvz, order='F')


def push_reference_forward(ref, U, Nx_e, Ny_e, Nz_e, h, xp=np):
    """Push the reference volume forward by the recovered displacement
    field onto the deformed-image grid: returns ref_pushed with
    ref_pushed(x) = ref(x - u(x)), where u(x) is the FE-interpolated
    displacement at voxel x (same interpolation as compute_active_mask),
    sampled via trilinear map_coordinates (mode='nearest'). For a converged
    correlation (g(x+u(x)) ~= f(x), see the residual definition in
    correlate_gpu), this is the small-deformation approximation
    g(x) ~= f(x-u(x)), so the result is on the same (Nvx,Nvy,Nvz) grid as
    deformed.npy and directly comparable to it.
    """
    return push_field_forward(ref, U, Nx_e, Ny_e, Nz_e, h, order=1, xp=xp)


def compute_overlap_composite(ref, pushed, p_lo=0.5, p_hi=99.5):
    """Build an (Nvx,Nvy,Nvz,3) uint8 RGB composite for visually checking
    the overlap between `ref` (reference image, shown in pink/magenta:
    R and B channels) and `pushed` (the forward-pushed reference image
    -- see push_reference_forward -- shown in green: G channel).

    Both volumes are clipped to the [p_lo, p_hi] percentile range of their
    combined intensities and rescaled to [0, 255] with the same shared
    scale, so brightness is directly comparable between channels. Regions
    where the two images align/overlap appear white/grey (pink + green);
    regions that only light up in `ref` appear pink, regions that only
    light up in `pushed` appear green.
    """
    ref = np.asarray(ref, dtype=np.float64)
    pushed = np.asarray(pushed, dtype=np.float64)

    combined = np.concatenate([ref.ravel(), pushed.ravel()])
    lo, hi = np.percentile(combined, [p_lo, p_hi])

    def to_u8(vol):
        scaled = (vol - lo) / (hi - lo)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    ref_u8 = to_u8(ref)
    pushed_u8 = to_u8(pushed)

    composite = np.zeros(ref.shape + (3,), dtype=np.uint8)
    composite[..., 0] = ref_u8     # R: reference -> pink
    composite[..., 1] = pushed_u8  # G: pushed reference -> green
    composite[..., 2] = ref_u8     # B: reference -> pink
    return composite


def compute_zncc(f_flat, g_warped_flat, mask_vox, xp=np):
    """Zero-normalized cross-correlation between the reference image and the
    deformed image warped by the recovered displacement, restricted to
    `mask_vox` (the active/overlap voxel set).

    ZNCC = sum((f-mean(f)) * (g-mean(g))) / sqrt(sum((f-mean(f))^2) * sum((g-mean(g))^2))

    Bounded in [-1, 1] (1 = perfect correlation), independent of image
    intensity scale/offset and of how many voxels are active -- unlike the
    raw residual std (grey levels, biased by active-voxel count), this makes
    it comparable across runs with different masks/meshes. This is the
    standard DIC/DVC correlation-quality metric, and (unlike ground-truth
    comparison) requires no known displacement field: it only needs `f`,
    `g` warped by the recovered `U`, and the active mask -- exactly what
    every run_dvc.py invocation already has. See CLAUDE.md's sensitivity
    sweep notes for why this alone should not be used to pick regularization
    strength (it is biased toward under-regularization, same as raw
    residual), but it is still the right per-run "how good is this
    correlation" figure when no ground truth exists.

    f_flat, g_warped_flat : (Nvox,) float arrays, same voxel ordering as
        mask_vox (e.g. ref.reshape(-1, order='F') and f_flat - res).
    mask_vox : (Nvox,) bool array -- voxels to include (e.g. active_vox).
    xp : numpy or cupy module (f_flat/g_warped_flat/mask_vox must match it).
    """
    f = xp.asarray(f_flat, dtype=xp.float64)[mask_vox]
    g = xp.asarray(g_warped_flat, dtype=xp.float64)[mask_vox]
    f = f - f.mean()
    g = g - g.mean()
    denom = xp.sqrt(xp.sum(f * f) * xp.sum(g * g))
    if denom <= 0:
        return float('nan')
    return float(xp.sum(f * g) / denom)


def write_outputs(OUTPUT_DIR, U, res, ref, deformed_for_mask, deformed_cropped,
                   Nx_e, Ny_e, Nz_e, h, g_origin=(0, 0, 0), U_push=None,
                   crop_lo=(0, 0, 0), xp=None, glt=0.0, dU=None):
    """Compute derived fields, print summary statistics, and save all
    `.npy` outputs to OUTPUT_DIR. See module docstring for the full list
    of files written.

    The voxel-scale derived fields -- the active/unsafe masks' driving
    `compute_active_mask`, the ZNCC proxy, and the two forward pushes -- are
    dominated by `interp_field_vox` + `map_coordinates` over full Nvox arrays
    and run ~50x faster on the GPU, so they are computed on `xp` (auto-cupy by
    default) and pulled back to host numpy only for saving and for the
    CPU-only stages. The element-scale stages -- the Hex8 principal strains,
    the porosity-aware unsafe/overlap-lost masks (kernels_dvc, intentionally
    numpy), and the scipy `griddata` deformed-config remaps (no cupy
    equivalent) -- stay on the CPU; they are ~h^3 smaller and not the
    bottleneck.

    Parameters
    ----------
    OUTPUT_DIR : str
    U : (Ndof,) flat nodal displacement array (numpy).
    res : (Nvox,) flat per-voxel residual array (numpy), Nvox=Nvx*Nvy*Nvz
        with Nvx,Nvy,Nvz = ref.shape.
    ref : (Nvx,Nvy,Nvz) reference volume defining the output region.
    deformed_for_mask : deformed volume passed to compute_active_mask
        (may be the full, uncropped deformed volume when g_origin != 0).
    deformed_cropped : (Nvx,Nvy,Nvz) deformed volume cropped to the same
        region as `ref` -- saved as deformed.npy and used to align
        the deformed-configuration remaps.
    Nx_e, Ny_e, Nz_e : int -- mesh element counts.
    h : int -- element edge length (voxels).
    g_origin : (3,) -- deformed-image origin relative to ref's local frame,
        passed to compute_active_mask (non-zero when def_preprocessed is offset from ref).
    glt : float, default 0.0 -- gray-level threshold; voxels with intensity
        <= glt (in ref, via mask0_vox = ref > glt, or in the sampled deformed
        image) are "no material / no contrast" and excluded from the active
        mask. Must match the value used by the correlation solve.
    xp : numpy or cupy module, or None (default) to auto-select cupy when a
        CUDA device is available (see _resolve_xp). Controls only which
        backend the voxel-scale stages run on; all inputs/outputs are numpy.
    dU : (Ndof,) flat nodal update array (numpy) or None (default). The final
        Gauss-Newton update returned by (multiscale_)correlate_gpu. When given,
        the per-cell convergence field gn_convergence_cell.npy (||dU||/||U|| per
        element) is computed and added to the .vti; omitted when None (older
        callers that don't thread dU through).
    """
    xp = _resolve_xp(xp)
    Nvx, Nvy, Nvz = ref.shape
    nbx, nby, nbz = Nx_e + 1, Ny_e + 1, Nz_e + 1

    # Reshape U into (Nx+1,Ny+1,Nz+1,3), axes (x,y,z), last axis (ux,uy,uz).
    # node = i + j*(Nx+1) + k*(Nx+1)*(Ny+1) (x fastest) -> Fortran order.
    def to_grid(comp):
        return comp.reshape(nbx, nby, nbz, order='F')

    U_recovered_grid = np.stack([to_grid(U[c::3]) for c in range(3)], axis=-1)

    # Per-voxel residual into (Nvx,Nvy,Nvz) (vidx = vx + vy*Nvx + vz*Nvx*Nvy,
    # x fastest -> Fortran order).
    residual_grid = res.reshape(Nvx, Nvy, Nvz, order='F')

    for c, name in enumerate(['ux', 'uy', 'uz']):
        comp = U_recovered_grid[..., c]
        print(f"  {name}: mean={comp.mean():.4f}  std={comp.std():.4f}  "
              f"min={comp.min():.4f}  max={comp.max():.4f}")

    n_active = int(np.count_nonzero(res))
    print(f"Nonzero (active) residual voxels: {n_active}/{res.size} "
          f"({100 * n_active / res.size:.1f}%)")
    print(f"residual std (active voxels): {res[res != 0].std():.4f}")

    # Unsafe-node mask: a node is "unsafe" if it belongs to an element that
    # contains at least one inactive voxel (outside the final dynamic mask).
    # unsafe_voxel_mask flags every voxel in an element that touches such a
    # node -- the one-element dilation of the lost-overlap region, i.e. the
    # voxels whose correlation result may be contaminated by regularization
    # rather than real image data (see CLAUDE.md mesh-border discussion).
    # mask0_vox: intensity <= glt in ref means "no material/no contrast" (see
    # CLAUDE.md) and is excluded from the correlation, same convention as
    # correlate_gpu's mask0_vox = f_vox > glt.
    mask0_vox = ref.reshape(-1, order='F') > glt
    # Voxel-scale GPU stages: the active mask and the ZNCC proxy. Run on xp
    # (auto-cupy), then pull `active_vox` back to host numpy for the CPU-only
    # kernels_dvc masks below.
    U_x = xp.asarray(U)
    active_vox_x = compute_active_mask(
        U_x, xp.asarray(deformed_for_mask), Nx_e, Ny_e, Nz_e, h,
        xp=xp, g_origin=g_origin, mask0_vox=xp.asarray(mask0_vox), glt=glt)

    # No-ground-truth accuracy proxy: ZNCC between f and g warped by the
    # recovered U, over the active voxel set (see compute_zncc docstring).
    f_flat_x = xp.asarray(ref).reshape(-1, order='F').astype(xp.float64)
    g_warped_x = f_flat_x - xp.asarray(res).astype(xp.float64)
    zncc = compute_zncc(f_flat_x, g_warped_x, active_vox_x, xp=xp)
    print(f"ZNCC (active voxels, no-ground-truth accuracy proxy): {zncc:.6f}")
    active_vox = _to_numpy(active_vox_x, xp)
    del f_flat_x, g_warped_x, active_vox_x
    unsafe_vox = compute_unsafe_voxel_mask(active_vox, h, Nx_e, Ny_e, Nz_e)
    unsafe_mask_grid = unsafe_vox.reshape(Nvx, Nvy, Nvz, order='F')
    n_unsafe = int(np.count_nonzero(unsafe_mask_grid))
    print(f"Unsafe voxels (affected by a node connected to an inactive voxel): "
          f"{n_unsafe}/{unsafe_mask_grid.size} ({100 * n_unsafe / unsafe_mask_grid.size:.1f}%)")

    # Per-element "safe" cell field for the .vti, reduced from the per-voxel
    # unsafe_voxel_mask (True where safe -- same inverse-of-unsafe convention as
    # view_reference.py's `~unsafe_voxel_mask` overlay).
    safe_elements_cell = compute_safe_cell_mask(unsafe_vox, h, Nx_e, Ny_e, Nz_e)
    n_safe = int(np.count_nonzero(safe_elements_cell))
    print(f"Safe elements (no unsafe voxel): {n_safe}/{safe_elements_cell.size} "
          f"({100 * n_safe / safe_elements_cell.size:.1f}%)")

    # Overlap-lost element mask (porosity-excluding), same (Nx,Ny,Nz) grid as
    # principal_strains_cell -- the mask to gate the cell-strain output. Unlike
    # the porosity-conflating unsafe_voxel_mask above, it flags an element only
    # if its *material* (mask0=True) lost correlation overlap OR the element
    # has too little material to measure strain (material count fraction < 0.5,
    # the min_material_frac default); well-correlated porosity is not flagged.
    # See compute_overlap_lost_element_mask for the full two-criterion def.
    overlap_lost_elem = compute_overlap_lost_element_mask(
        active_vox, mask0_vox, h, Nx_e, Ny_e, Nz_e)
    overlap_lost_element_mask = overlap_lost_elem.reshape(Nx_e, Ny_e, Nz_e, order='F')
    n_overlap_lost = int(np.count_nonzero(overlap_lost_element_mask))
    print(f"Overlap-lost elements (lost material overlap or <50% material): "
          f"{n_overlap_lost}/{overlap_lost_element_mask.size} "
          f"({100 * n_overlap_lost / overlap_lost_element_mask.size:.1f}%)")

    print(f"\ndeformed image: mean={deformed_cropped.mean():.2f}  "
          f"min={deformed_cropped.min():.2f}  max={deformed_cropped.max():.2f}")

    # Cell-centered principal strains, via Hex8 shape-function gradients
    # evaluated at each element's centroid (see compute_principal_strains_cell).
    principal_strains_cell = compute_principal_strains_cell(U_recovered_grid, h)
    for c, name in enumerate(['e1 (max)', 'e2 (mid)', 'e3 (min)']):
        comp = principal_strains_cell[..., c]
        print(f"  cell {name}: mean={comp.mean():.5f}  std={comp.std():.5f}  "
              f"min={comp.min():.5f}  max={comp.max():.5f}")

    principal_strains_cell_deformed = remap_cell_field_to_deformed(
        principal_strains_cell, U_recovered_grid, h, crop_lo=crop_lo)
    for c, name in enumerate(['e1 (max)', 'e2 (mid)', 'e3 (min)']):
        comp = principal_strains_cell_deformed[..., c]
        print(f"  deformed-config cell {name}: mean={comp.mean():.5f}  std={comp.std():.5f}  "
              f"min={comp.min():.5f}  max={comp.max():.5f}")

    # Von Mises equivalent strain (single deviatoric-magnitude scalar per
    # element) derived from the principal strains, in both the reference and
    # deformed configurations (see compute_equivalent_strain). The deformed
    # field is computed from principal_strains_cell_deformed so it is exactly
    # the equivalent strain of the field visualized on the deformed grid.
    equivalent_strain_cell = compute_equivalent_strain(principal_strains_cell)
    equivalent_strain_cell_deformed = compute_equivalent_strain(
        principal_strains_cell_deformed)
    print(f"  cell eq. strain: mean={equivalent_strain_cell.mean():.5f}  "
          f"std={equivalent_strain_cell.std():.5f}  "
          f"min={equivalent_strain_cell.min():.5f}  "
          f"max={equivalent_strain_cell.max():.5f}")
    print(f"  deformed-config cell eq. strain: "
          f"mean={equivalent_strain_cell_deformed.mean():.5f}  "
          f"std={equivalent_strain_cell_deformed.std():.5f}  "
          f"min={equivalent_strain_cell_deformed.min():.5f}  "
          f"max={equivalent_strain_cell_deformed.max():.5f}")

    # Per-cell convergence / correlation-quality fields.
    #  (1) residual_std_cell + zncc_cell: how well the recovered displacement
    #      fits the images inside each element (from the final residual).
    #  (2) gn_convergence_cell: ||dU||/||U|| per element, the local analog of
    #      the global dU/U the solve stops on (only when dU is threaded through).
    residual_std_cell, zncc_cell = compute_correlation_quality_cell(
        res, ref, h, Nx_e, Ny_e, Nz_e)
    print(f"  cell residual std: mean={np.nanmean(residual_std_cell):.4f}  "
          f"min={np.nanmin(residual_std_cell):.4f}  "
          f"max={np.nanmax(residual_std_cell):.4f}")
    print(f"  cell ZNCC: mean={np.nanmean(zncc_cell):.5f}  "
          f"min={np.nanmin(zncc_cell):.5f}  max={np.nanmax(zncc_cell):.5f}")

    gn_convergence_cell = None
    if dU is not None:
        gn_convergence_cell = compute_gn_convergence_cell(dU, U, Nx_e, Ny_e, Nz_e)
        print(f"  cell GN convergence dU/U: mean={np.nanmean(gn_convergence_cell):.2e}  "
              f"min={np.nanmin(gn_convergence_cell):.2e}  "
              f"max={np.nanmax(gn_convergence_cell):.2e}")

    # Overlap-lost element mask pushed to the deformed config (same nearest-
    # neighbour element-center remap as principal_strains_cell_deformed), so
    # it gates the deformed-configuration cell strain.
    overlap_lost_element_mask_deformed = remap_element_mask_to_deformed(
        overlap_lost_element_mask, U_recovered_grid, h, crop_lo=crop_lo)
    n_overlap_lost_def = int(np.count_nonzero(overlap_lost_element_mask_deformed))
    print(f"Overlap-lost elements (deformed config): {n_overlap_lost_def}/"
          f"{overlap_lost_element_mask_deformed.size} "
          f"({100 * n_overlap_lost_def / overlap_lost_element_mask_deformed.size:.1f}%)")

    # U_push is the displacement used for the forward push (ref -> deformed).
    # Callers that pre-include a large rigid-body / affine component in U should
    # pass U_push = U_local (the small residual after subtracting the affine) so
    # that ref(x - u_local) stays within the ref domain.  When U_push is None
    # (default) the full U is used, which is correct for the standard case where
    # U is already a small, purely local displacement.
    U_fwd_x = xp.asarray(U_push) if U_push is not None else U_x
    ref_forward_pushed = _to_numpy(
        push_reference_forward(xp.asarray(ref), U_fwd_x, Nx_e, Ny_e, Nz_e, h, xp=xp), xp)
    diff = ref_forward_pushed.astype(np.float64) - deformed_cropped.astype(np.float64)
    print(f"\nref_forward_pushed - deformed_cropped: mean={diff.mean():.4f}  "
          f"std={diff.std():.4f}  min={diff.min():.4f}  max={diff.max():.4f}")

    # Forward-pushed unsafe_voxel_mask: same push as ref_forward_pushed, but
    # nearest-neighbour (order=0) since the field is boolean, and
    # mode='constant', cval=1 so that voxels whose source falls outside the
    # reference domain are flagged unsafe rather than inheriting a
    # clamped edge value (see push_field_forward docstring).
    unsafe_voxel_mask_forward_pushed = _to_numpy(push_field_forward(
        xp.asarray(unsafe_mask_grid), U_fwd_x, Nx_e, Ny_e, Nz_e, h, order=0,
        mode='constant', cval=1, xp=xp), xp).astype(bool)
    n_unsafe_pushed = int(np.count_nonzero(unsafe_voxel_mask_forward_pushed))
    print(f"Unsafe voxels (forward-pushed to deformed grid): {n_unsafe_pushed}/"
          f"{unsafe_voxel_mask_forward_pushed.size} "
          f"({100 * n_unsafe_pushed / unsafe_voxel_mask_forward_pushed.size:.1f}%)")

    np.save(os.path.join(OUTPUT_DIR, 'U_recovered.npy'), U_recovered_grid)
    np.save(os.path.join(OUTPUT_DIR, 'residual.npy'), residual_grid)
    np.save(os.path.join(OUTPUT_DIR, 'principal_strains_cell.npy'), principal_strains_cell)
    np.save(os.path.join(OUTPUT_DIR, 'principal_strains_cell_deformed.npy'), principal_strains_cell_deformed)
    np.save(os.path.join(OUTPUT_DIR, 'equivalent_strain_cell.npy'), equivalent_strain_cell)
    np.save(os.path.join(OUTPUT_DIR, 'equivalent_strain_cell_deformed.npy'), equivalent_strain_cell_deformed)
    np.save(os.path.join(OUTPUT_DIR, 'residual_std_cell.npy'), residual_std_cell)
    np.save(os.path.join(OUTPUT_DIR, 'zncc_cell.npy'), zncc_cell)
    if gn_convergence_cell is not None:
        np.save(os.path.join(OUTPUT_DIR, 'gn_convergence_cell.npy'), gn_convergence_cell)
    np.save(os.path.join(OUTPUT_DIR, 'ref.npy'), ref) # reference image
    np.save(os.path.join(OUTPUT_DIR, 'deformed.npy'), deformed_cropped) # deformed image
    np.save(os.path.join(OUTPUT_DIR, 'unsafe_voxel_mask.npy'), unsafe_mask_grid)
    np.save(os.path.join(OUTPUT_DIR, 'overlap_lost_element_mask.npy'), overlap_lost_element_mask)
    np.save(os.path.join(OUTPUT_DIR, 'overlap_lost_element_mask_deformed.npy'),
            overlap_lost_element_mask_deformed)
    np.save(os.path.join(OUTPUT_DIR, 'ref_forward_pushed.npy'), ref_forward_pushed)
    np.save(os.path.join(OUTPUT_DIR, 'unsafe_voxel_mask_forward_pushed.npy'), unsafe_voxel_mask_forward_pushed)

    # ParaView/VisIt-ready binary XML ImageData (.vti): nodal displacements as
    # POINT data and the principal + equivalent strains as CELL data on one
    # mesh-resolution grid. spacing=(h,h,h) puts node positions in voxel units
    # so the voxel-unit displacement warps correctly. Guarded: a missing/broken
    # vtk_export must never fail an otherwise-complete correlation run.
    # Per-cell material mask: True where the element's mean reference gray
    # level exceeds glt (bundled into the .vti as a cell field to gate the
    # strains -- e.g. ParaView's Threshold filter).
    material_cell_mask = compute_material_cell_mask(
        ref, h, Nx_e, Ny_e, Nz_e, glt=glt)
    n_material = int(np.count_nonzero(material_cell_mask))
    print(f"Material cells (mean reference gray > glt): {n_material}/"
          f"{material_cell_mask.size} "
          f"({100 * n_material / material_cell_mask.size:.1f}%)")

    # Eroded material mask: material cells with no non-material neighbour (drops
    # the outer element layer bordering pores/background), for gating strains
    # away from the unreliable material/void interface.
    material_cell_mask_eroded = compute_material_cell_mask_eroded(material_cell_mask)
    n_material_eroded = int(np.count_nonzero(material_cell_mask_eroded))
    print(f"Material cells (eroded, no non-material neighbour): {n_material_eroded}/"
          f"{material_cell_mask_eroded.size} "
          f"({100 * n_material_eroded / material_cell_mask_eroded.size:.1f}%)")
    np.save(os.path.join(OUTPUT_DIR, 'material_cell_mask.npy'), material_cell_mask)
    np.save(os.path.join(OUTPUT_DIR, 'material_cell_mask_eroded.npy'), material_cell_mask_eroded)

    # Per-cell convergence / quality fields go into the .vti as CELL scalars
    # alongside the strains and material mask, so they gate/colour the same
    # ParaView grid. NaN (inactive/undefined elements) is written as-is.
    extra_cell = {
        'material_mask': material_cell_mask,
        'material_mask_eroded': material_cell_mask_eroded,
        'safe_elements': safe_elements_cell,
        'residual_std': residual_std_cell,
        'zncc': zncc_cell,
    }
    if gn_convergence_cell is not None:
        extra_cell['gn_convergence'] = gn_convergence_cell

    try:
        vtk_export = _load_vtk_export()
        if vtk_export is None:
            print("\nvtk_export module not importable -- skipped dvc_fields.vti export")
        else:
            vti_path = os.path.join(OUTPUT_DIR, 'dvc_fields.vti')
            vtk_export.save_dvc_fields_vti(
                vti_path, principal_strains_cell, equivalent_strain_cell,
                U_recovered_grid, spacing=(h, h, h),
                extra_cell=extra_cell)
            print(f"\nSaved dvc_fields.vti (eps1..eps3, eps_eq, material_mask, "
                  f"material_mask_eroded, safe_elements, residual_std, zncc"
                  f"{', gn_convergence' if gn_convergence_cell is not None else ''} "
                  f"cell data; displacement point data) -> {vti_path}")
    except Exception as e:
        print(f"\nWARNING: dvc_fields.vti export failed ({e!r}); "
              f".npy outputs are unaffected")

    print(f"\nSaved U_recovered.npy {U_recovered_grid.shape}, "
          f"residual.npy {residual_grid.shape}, "
          f"residual_std_cell.npy {residual_std_cell.shape}, "
          f"zncc_cell.npy {zncc_cell.shape}, "
          + (f"gn_convergence_cell.npy {gn_convergence_cell.shape}, "
             if gn_convergence_cell is not None else "") +
          f"principal_strains_cell.npy {principal_strains_cell.shape}, "
          f"principal_strains_cell_deformed.npy {principal_strains_cell_deformed.shape}, "
          f"equivalent_strain_cell.npy {equivalent_strain_cell.shape}, "
          f"equivalent_strain_cell_deformed.npy {equivalent_strain_cell_deformed.shape}, "
          f"ref.npy {ref.shape}, "
          f"deformed.npy {deformed_cropped.shape}, "
          f"unsafe_voxel_mask.npy {unsafe_mask_grid.shape}, "
          f"overlap_lost_element_mask.npy {overlap_lost_element_mask.shape}, "
          f"overlap_lost_element_mask_deformed.npy {overlap_lost_element_mask_deformed.shape}, "
          f"ref_forward_pushed.npy {ref_forward_pushed.shape} and "
          f"unsafe_voxel_mask_forward_pushed.npy {unsafe_voxel_mask_forward_pushed.shape} -> {OUTPUT_DIR}")
