# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-18 20:11:45
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-18 21:32:10

"""End-to-end matrix-free GPU DVC correlation loop (Phase 2).

Assumes:

  - A structured Hex8 mesh with Nx*Ny*Nz elements of integer edge length
    `h` (voxel-center quadrature, see geometry_dvc.build_N_stencil), built
    with mesh.Connectivity(order='N') so DOF ordering matches
    kernels_dvc's implicit connectivity.
  - scale = 1: mesh coordinates coincide with image voxel coordinates, so
    the (h+1)^3 quadrature points per element fall exactly on integer voxel
    centers, vidx = vx + vy*Nvx + vz*Nvx*Nvy (Nvx=Nx*h+1, Nvy=Ny*h+1).
  - The reference image `f.pix` has shape exactly (Nx*h+1, Ny*h+1, Nz*h+1)
    (the mesh spans the whole reference image) -- so mask0 is True
    everywhere and the reference-side gradient/value arrays are obtained
    directly from f.pix via xp.gradient, without per-quad-point
    interpolation.

The reference-image gradient via xp.gradient (central differences, 2nd
order one-sided at the array boundary) differs slightly from
image.Volume.InterpGrad's finite-difference-of-the-trilinear-interpolant
at boundary voxels (1st order there); this is a frozen-H approximation
and does not affect convergence, only the GN curvature estimate.
"""
import numpy as np

from .geometry_dvc import build_N_stencil
from .kernels_dvc import (
    get_elem_dofs, get_voxel_offsets, get_elem_voxel_origins,
    matvec_H, matvec_K_ref, build_diagonal_H, build_diagonal_K_ref, pcg,
    auto_chunk_size,
    matvec_equilibrium_gap, build_diagonal_equilibrium_gap,
)


def _map_coordinates(pix, coords, xp, order=1, mode='nearest', cval=0.0):
    if xp is np:
        from scipy.ndimage import map_coordinates
    else:
        from cupyx.scipy.ndimage import map_coordinates
    return map_coordinates(pix, coords, order=order, mode=mode, cval=cval)


def image_grad_and_values(pix, xp, dtype=np.float32):
    """Reference-image values and gradient at every voxel, flattened with
    vidx = vx + vy*Nvx + vz*Nvx*Nvy (x fastest), matching kernels_dvc's
    voxel indexing convention.

    Returns
    -------
    f_vox : (Nvox,) array
    grad_f_vox : (Nvox, 3) array
    """
    pix_x = xp.asarray(pix, dtype=dtype)
    f_vox = xp.ravel(pix_x, order='F')
    # Fill grad_f_vox one axis at a time into a preallocated (Nvox, 3) buffer.
    # The equivalent xp.stack([ravel(gx), ravel(gy), ravel(gz)], axis=-1)
    # momentarily holds all three gradient components AND their three F-order
    # ravel copies AND the stack output simultaneously (~9 full Nvox arrays);
    # on GPU those transient blocks are then reserved by the CuPy pool for the
    # whole run. Computing/raveling one component at a time keeps the extra
    # footprint to ~2 Nvox arrays. Bit-identical: grad_f_vox[:, c] here equals
    # ravel(gradient(pix)[c], 'F') = the old stack's c-th column.
    grad_f_vox = xp.empty((pix_x.size, 3), dtype=dtype)
    for c in range(3):
        grad_f_vox[:, c] = xp.ravel(xp.gradient(pix_x, axis=c), order='F')
    return f_vox, grad_f_vox


def erode_mask0_vox(mask0_vox, Nvx, Nvy, Nvz, xp):
    """Erode a per-voxel reference-validity mask by one voxel along each
    axis (6-connectivity / face-neighbours), matching xp.gradient's central-
    difference stencil: a voxel is excluded if it or any of its 6 face
    neighbours is invalid (e.g. zero-intensity/no-material, see mask0_vox in
    correlate_gpu). Used to zero grad_f_vox at the material/air interface,
    where xp.gradient would otherwise produce a large spurious gradient from
    the intensity step rather than real material texture.
    """
    if xp is np:
        from scipy.ndimage import minimum_filter
    else:
        from cupyx.scipy.ndimage import minimum_filter
    grid = mask0_vox.reshape(Nvx, Nvy, Nvz, order='F')
    footprint = xp.zeros((3, 3, 3), dtype=bool)
    footprint[1, 1, :] = True
    footprint[1, :, 1] = True
    footprint[:, 1, 1] = True
    eroded = minimum_filter(grid, footprint=footprint, mode='nearest')
    return xp.ravel(eroded, order='F')


def gather_quadpoint_values(field_vox, h, Nx, Ny, Nz, xp):
    """Replicate a per-voxel scalar field onto the (Ne*(h+1)^3,) array of
    quadrature-point values (one entry per (element, stencil point), with
    the same multiplicity as mesh.DVCIntegrationVoxel's per-element
    assembly loop / dic.py's self.f, self.mask0, etc.). Needed to reproduce
    dic.py's mean()/std() normalization, which is taken over this
    multiplicity-weighted array, not over unique voxels.
    """
    Ne = Nx * Ny * Nz
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, xp)
    elem_ids = xp.arange(Ne, dtype=xp.int32)
    vox0 = get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp)
    vox_ids = vox0[:, None] + voxel_offsets[None, :]  # (Ne, nq)
    return field_vox[vox_ids.reshape(-1)]


def masked_quadpoint_mean_std(field_vox, mask_vox, h, Nx, Ny, Nz, xp,
                              chunk_size=None):
    """Mean and population std of `field_vox` over the multiplicity-weighted
    quadrature-point array, restricted to `mask_vox`, without materializing
    the full (Ne*(h+1)^3,) gather array.

    Equivalent to (but bounded to a (chunk, nq) working set):
        qp = gather_quadpoint_values(field_vox, ...)
        m  = gather_quadpoint_values(mask_vox,  ...)
        mean, std = qp[m].mean(), qp[m].std()   # numpy ddof=0

    Count / sum / sum-of-squares are accumulated in float64, so the result is
    at least as accurate as the float32 full-array reduction it replaces (rel.
    diff ~1e-8, below the GPU solve's run-to-run add.at noise floor ~4e-8); it
    is NOT bit-identical to that reduction. Chunking here removes the dominant
    per-iteration VRAM transient: each gather_quadpoint_values call otherwise
    builds a full (Ne, (h+1)^3) int32 index tensor plus its output (~2.3 GB
    combined on a ~2e8-voxel volume).
    """
    Ne = Nx * Ny * Nz
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, xp)
    nq = voxel_offsets.shape[0]
    if chunk_size is None:
        chunk_size = auto_chunk_size(nq)

    count = xp.zeros((), dtype=xp.float64)
    s1 = xp.zeros((), dtype=xp.float64)
    s2 = xp.zeros((), dtype=xp.float64)
    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        vox0 = get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp)
        vox_ids = vox0[:, None] + voxel_offsets[None, :]      # (chunk, nq)
        mf = mask_vox[vox_ids].astype(xp.float64)             # (chunk, nq)
        vals = field_vox[vox_ids].astype(xp.float64) * mf     # zeroed outside mask
        count += mf.sum()
        s1 += vals.sum()
        s2 += (vals * vals).sum()

    c = float(count)
    if c < 1:
        return 0.0, 0.0
    mean = float(s1) / c
    var = float(s2) / c - mean * mean
    return mean, (var if var > 0.0 else 0.0) ** 0.5


def interp_field_vox(U, N_stencil, h, Nx, Ny, Nz, xp, chunk_size=200_000,
                     channels_first=False):
    """Displacement field (3 components) evaluated at every voxel center,
    f_vox[vidx, c] = sum_a N_stencil(q, a) * U[dof(e, a, c)] -- the
    matrix-free equivalent of m.phix/phiy/phiz @ U at voxel-center quad
    points.

    Returns a (Nvox, 3) array by default, or a (3, Nvox) channels-first array
    if `channels_first=True` -- the latter is what map_coordinates wants as
    its coordinate argument, so correlate_gpu can build the deformed sample
    coordinates straight into this layout (adding the voxel ramp in place)
    without a separate (Nvox, 3) displacement array or an xp.stack copy.
    """
    Ne = Nx * Ny * Nz
    Nvx, Nvy, Nvz = Nx * h + 1, Ny * h + 1, Nz * h + 1
    Nvox = Nvx * Nvy * Nvz

    N_stencil_x = xp.asarray(N_stencil, dtype=U.dtype)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, xp)
    nq = voxel_offsets.shape[0]

    out = xp.zeros((3, Nvox) if channels_first else (Nvox, 3), dtype=U.dtype)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        dofs = get_elem_dofs(elem_ids, Nx, Ny, xp)          # (chunk, 24)
        u_loc = U[dofs].reshape(-1, 8, 3)                    # (chunk, 8, 3)

        vox0 = get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp)
        vox_ids = vox0[:, None] + voxel_offsets[None, :]     # (chunk, nq)

        vals = xp.einsum('qa,kac->kqc', N_stencil_x, u_loc)  # (chunk, nq, 3)
        if channels_first:
            out[:, vox_ids.reshape(-1)] = vals.reshape(-1, 3).T
        else:
            out[vox_ids.reshape(-1)] = vals.reshape(-1, 3)

    return out


def sample_deformed_field(U, g_pix_x, N_stencil, h, Nx, Ny, Nz, xp,
                          g_origin=(0, 0, 0), mask0_vox=None, chunk_size=None,
                          glt=0.0):
    """Sample the deformed image g at every voxel's deformed position
    x + u(x), plus the per-voxel active mask (in_bounds & mask0 & g>glt), in
    element chunks -- so the full (3, Nvox) deformed-coordinate array is never
    materialized. That array is the largest per-GN-iteration buffer (12*Nvox
    bytes, ~2.5 GB at 2e8 voxels) and sets the sampling-stage VRAM peak.

    Bit-identical to building the full coordinate array and calling
    map_coordinates once: interp (einsum) and order-1 sampling are point-wise,
    u is single-valued at element-shared voxels, and the per-element scatter
    order (highest elem_id touching a voxel wins) matches interp_field_vox's,
    so the scattered g_vals/mask_vox come out the same.

    Returns
    -------
    g_vals : (Nvox,) array -- g(x+u(x)) at every voxel (boundary-clamped where
        x+u leaves g's bounds, matching the un-chunked map_coordinates path).
    mask_vox : (Nvox,) bool array -- in_bounds(x+u) & mask0_vox & (g>glt).

    glt : float, default 0.0
        Gray-level threshold below which a sampled deformed voxel is "no
        material / no contrast" and dropped from the active mask.
    """
    Ne = Nx * Ny * Nz
    Nvx, Nvy, Nvz = Nx * h + 1, Ny * h + 1, Nz * h + 1
    Nvox = Nvx * Nvy * Nvz
    NxNy = Nvx * Nvy
    gnx, gny, gnz = g_pix_x.shape
    gx0, gy0, gz0 = g_origin
    dtype = U.dtype

    if mask0_vox is None:
        mask0_vox = xp.ones(Nvox, dtype=bool)

    N_stencil_x = xp.asarray(N_stencil, dtype=dtype)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, xp)
    nq = voxel_offsets.shape[0]
    if chunk_size is None:
        chunk_size = max(1, 4_000_000 // nq)

    g_vals = xp.zeros(Nvox, dtype=dtype)
    mask_vox = xp.zeros(Nvox, dtype=bool)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        dofs = get_elem_dofs(elem_ids, Nx, Ny, xp)             # (chunk, 24)
        u_loc = U[dofs].reshape(-1, 8, 3)                      # (chunk, 8, 3)
        vox0 = get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp)
        vox_ids = vox0[:, None] + voxel_offsets[None, :]       # (chunk, nq)

        u_qp = xp.einsum('qa,kac->kqc', N_stencil_x, u_loc)    # (chunk, nq, 3)

        # Voxel ramp x decoded from the flat quad-point voxel index (x fastest,
        # vidx = vx + vy*Nvx + vz*Nvx*Nvy). Same (u + x) - g_origin order as the
        # un-chunked path.
        vx = (vox_ids % Nvx).astype(dtype)
        vy = ((vox_ids // Nvx) % Nvy).astype(dtype)
        vz = (vox_ids // NxNy).astype(dtype)
        cx = u_qp[:, :, 0] + vx - gx0
        cy = u_qp[:, :, 1] + vy - gy0
        cz = u_qp[:, :, 2] + vz - gz0

        in_bounds = ((cx >= 0) & (cx <= gnx - 1) &
                     (cy >= 0) & (cy <= gny - 1) &
                     (cz >= 0) & (cz <= gnz - 1))
        xp.clip(cx, 0, gnx - 1, out=cx)
        xp.clip(cy, 0, gny - 1, out=cy)
        xp.clip(cz, 0, gnz - 1, out=cz)

        coords = xp.stack([cx.reshape(-1), cy.reshape(-1), cz.reshape(-1)])
        g_chunk = _map_coordinates(g_pix_x, coords, xp, order=1).reshape(cx.shape)

        m = in_bounds & mask0_vox[vox_ids] & (g_chunk > glt)
        flat = vox_ids.reshape(-1)
        g_vals[flat] = g_chunk.reshape(-1)
        mask_vox[flat] = m.reshape(-1)

    return g_vals, mask_vox


def compute_active_mask(U, g_pix, Nx, Ny, Nz, h, xp, g_origin=(0, 0, 0),
                        mask0_vox=None, glt=0.0):
    """Per-voxel active mask (mask0 & g.InBounds(x+u) & g(x+u)>glt) for a
    displacement field U, reproducing the dynamic mask computed inside
    correlate_gpu's GN loop (lines computing `mask_vox`) for a
    converged/given U.

    glt : float, default 0.0
        Gray-level threshold below which a sampled deformed voxel counts as
        "no material/no contrast" and is excluded from the mask.

    Returns
    -------
    mask_vox : (Nvox,) bool array, True where the voxel's deformed position
        x+u(x) falls inside g_pix with intensity above glt there (and
        mask0_vox, if given) -- intensity <= glt denotes "no material/no
        contrast" (see CLAUDE.md) and is excluded from both f and g.
    """
    Nvx, Nvy, Nvz = Nx * h + 1, Ny * h + 1, Nz * h + 1
    Nvox = Nvx * Nvy * Nvz

    if mask0_vox is None:
        mask0_vox = xp.ones(Nvox, dtype=bool)

    N_stencil, _ = build_N_stencil(h)
    u_vox = interp_field_vox(U, N_stencil, h, Nx, Ny, Nz, xp)

    # Deformed voxel positions x+u, adding the voxel ramp x (x fastest,
    # vidx = vx + vy*Nvx + vz*Nvx*Nvy) in place through a Fortran-reshaped view
    # of each displacement component, so the three full (Nvox,) ramps are never
    # materialized (mirrors correlate_gpu's GN loop).
    ax = xp.arange(Nvx, dtype=U.dtype)
    ay = xp.arange(Nvy, dtype=U.dtype)
    az = xp.arange(Nvz, dtype=U.dtype)
    pgu = u_vox[:, 0].copy()
    pgv = u_vox[:, 1].copy()
    pgw = u_vox[:, 2].copy()
    pgu.reshape(Nvx, Nvy, Nvz, order='F')[...] += ax[:, None, None]
    pgv.reshape(Nvx, Nvy, Nvz, order='F')[...] += ay[None, :, None]
    pgw.reshape(Nvx, Nvy, Nvz, order='F')[...] += az[None, None, :]

    gx0, gy0, gz0 = g_origin
    g_pix_x = xp.asarray(g_pix)
    gnx, gny, gnz = g_pix_x.shape
    cx, cy, cz = pgu - gx0, pgv - gy0, pgw - gz0
    in_bounds = ((cx >= 0) & (cx <= gnx - 1) &
                 (cy >= 0) & (cy <= gny - 1) &
                 (cz >= 0) & (cz <= gnz - 1))

    xp.clip(cx, 0, gnx - 1, out=cx)
    xp.clip(cy, 0, gny - 1, out=cy)
    xp.clip(cz, 0, gnz - 1, out=cz)
    g_vals = _map_coordinates(g_pix_x, xp.stack([cx, cy, cz]), xp, order=1)

    return in_bounds & mask0_vox & (g_vals > glt)


def compute_b_gpu(res_vox, mask_vox, grad_f_vox, N_stencil, w_stencil, h,
                  Nx, Ny, Nz, Ndof, xp, chunk_size=None):
    """b = phiJdf.T @ diag(wdetJ * mask) @ res, matrix-free:
    b_local_e = sum_q w_q * mask_q * res_q * v_eq, with
    v_eq = N_stencil(q) (x) grad_f(voxel q) (same per-element vector as in
    matvec_H).
    """
    f = xp.zeros(Ndof, dtype=grad_f_vox.dtype)
    Ne = Nx * Ny * Nz

    N_stencil_x = xp.asarray(N_stencil, dtype=grad_f_vox.dtype)
    w_stencil_x = xp.asarray(w_stencil, dtype=grad_f_vox.dtype)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, xp)
    nq = voxel_offsets.shape[0]
    if chunk_size is None:
        chunk_size = auto_chunk_size(nq)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        dofs = get_elem_dofs(elem_ids, Nx, Ny, xp)               # (chunk, 24)

        vox0 = get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp)
        vox_ids = vox0[:, None] + voxel_offsets[None, :]          # (chunk, nq)

        gradf_q = grad_f_vox[vox_ids]                             # (chunk, nq, 3)
        res_q = res_vox[vox_ids]                                  # (chunk, nq)
        mask_q = mask_vox[vox_ids]                                # (chunk, nq)

        # b_local[c,3a+k] = sum_q (w_q mask_q res_q) N[q,a] gradf[c,q,k],
        # separable -- no (chunk, nq, 24) v_eq tensor (cf. matvec_H).
        coeff = w_stencil_x[None, :] * mask_q * res_q             # (chunk, nq)
        tmp = coeff[:, :, None] * gradf_q                         # (chunk, nq, 3)
        b_loc = xp.einsum('qa,cqk->cak', N_stencil_x, tmp).reshape(-1, 24)

        xp.add.at(f, dofs.ravel(), b_loc.ravel())

    return f


def fft_rigid_shift(f_pix, g_pix, xp):
    """Estimate a rigid voxel-shift d=(dx,dy,dz) via FFT phase correlation,
    in the convention g(x) ~= f(x - d) used throughout correlate_gpu (so d
    is directly usable as a uniform initial displacement U0).

    f_pix and g_pix are cropped to their common shape (from index 0, i.e.
    the shared corner) before the FFT if their shapes differ.

    Returns
    -------
    d : (3,) numpy array of floats (integer-valued voxel shifts).
    """
    f_x = xp.asarray(f_pix) #, dtype=xp.float64)
    g_x = xp.asarray(g_pix) #, dtype=xp.float64)
    shape = tuple(min(a, b) for a, b in zip(f_x.shape, g_x.shape))
    slices = tuple(slice(0, n) for n in shape)
    f_c = f_x[slices]
    g_c = g_x[slices]

    F = xp.fft.fftn(f_c)
    G = xp.fft.fftn(g_c)
    R = F * xp.conj(G)
    R = R / (xp.abs(R) + 1e-12)
    r = xp.fft.ifftn(R).real
    peak = xp.unravel_index(xp.argmax(r), r.shape)

    # ifftn(F*conj(G)/|.|) peaks at -d for g(x) = f(x - d).
    d = []
    for p, n in zip(peak, shape):
        p = int(p)
        if p > n // 2:
            p -= n
        d.append(-float(p))
    return np.array(d, dtype=np.float64)


def _zncc_masked(f_flat, g_samp, valid, xp):
    """Zero-normalized cross-correlation between f_flat and g_samp,
    restricted to `valid` (bool mask). Returns 1-ZNCC (cost to minimize,
    range [0, 2]), or 2.0 if too few valid points / degenerate variance.
    """
    n_valid = int(valid.sum())
    if n_valid < 1000:
        return 2.0
    fv = f_flat[valid]
    gv = g_samp[valid]
    fv = fv - fv.mean()
    gv = gv - gv.mean()
    denom = xp.sqrt((fv * fv).sum() * (gv * gv).sum())
    if float(denom) < 1e-8:
        return 2.0
    zncc = float((fv * gv).sum() / denom)
    return 1.0 - zncc


def _build_affine_matrix(p):
    """A = R(rx,ry,rz) @ diag(sx,sy,sz) from p = (rx,ry,rz,sx,sy,sz,...)."""
    rx, ry, rz = p[0], p[1], p[2]
    sx, sy, sz = p[3], p[4], p[5]
    cx, sxr = np.cos(rx), np.sin(rx)
    cy, syr = np.cos(ry), np.sin(ry)
    cz, szr = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sxr], [0, sxr, cx]])
    Ry = np.array([[cy, 0, syr], [0, 1, 0], [-syr, 0, cy]])
    Rz = np.array([[cz, -szr, 0], [szr, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    S = np.diag([sx, sy, sz])
    return R @ S


def affine_prealign(f_pix, g_pix, xp, decimate=4, t0=None,
                     angle_bound=0.3, scale_bound=0.15, maxiter=200,
                     disp=False):
    """Estimate a 9-DOF affine alignment (rotation + independent per-axis
    scale + translation) between f_pix and g_pix by minimizing 1-ZNCC
    (zero-normalized cross-correlation) on a decimated copy of the volumes.

    Models the warp x' = A(x - c) + c + t, with A = R(rx,ry,rz) @
    diag(sx,sy,sz) (3 Euler angles, 3 independent axis scale factors, no
    shear), c the volume center, and t a translation -- the same convention
    fft_rigid_shift uses for a pure-translation U0 (g(x) ~= f(x')). The
    returned A/t/c can be used to build a uniform affine U0:
    u(x) = A @ (x - c) + c + t - x.

    The ZNCC is computed over the *full* in-bounds intensity field, NOT
    restricted to material-on-material voxels. For low-contrast specimens
    (e.g. PA6GF30, whose material interior is nearly flat -- CoV ~0.19) the
    registration signal lives almost entirely in the material/pore boundary
    geometry; masking to f>0 & g>0 (as the main DVC solver does, where a
    zero gradient carries no signal) deletes exactly that boundary and
    leaves only interior noise (ZNCC ~0 even at the true alignment), so the
    search cannot find a gradient. Treating zero-intensity (pore) voxels as
    legitimate low-intensity contrast restores ZNCC ~0.86 at identity. Only
    samples whose warped position x' falls outside g's bounds are excluded.

    Parameters
    ----------
    f_pix, g_pix : array (Nx,Ny,Nz)
        Reference / deformed volumes (numpy or cupy, matching xp).
    decimate : int, default 4
        Stride used to subsample both volumes before the search -- keeps
        each cost-function evaluation cheap (a handful of ms on GPU).
    t0 : (3,) array, optional
        Initial translation guess in full-resolution voxel units (e.g. from
        fft_rigid_shift). If None, fft_rigid_shift is run on the decimated
        volumes to seed the translation.
    angle_bound : float, default 0.3
        Search bound on each Euler angle (radians, ~17 deg).
    scale_bound : float, default 0.15
        Search bound on |s_i - 1| for each per-axis scale factor.
    maxiter : int, default 200
        Passed to scipy.optimize.minimize (method='Powell').

    Returns
    -------
    A : (3,3) ndarray
    t : (3,) ndarray, translation in full-resolution voxel units
    c : (3,) ndarray, center of rotation/scaling in full-resolution voxel
        units
    info : dict with keys 'cost_identity', 'cost_translation_only',
        'cost_final', 'nfev', 'p' (raw 9-parameter optimum, decimated units).
    """
    from scipy.optimize import minimize

    f_d = xp.asarray(f_pix, dtype=xp.float32)[::decimate, ::decimate, ::decimate]
    g_d = xp.asarray(g_pix, dtype=xp.float32)[::decimate, ::decimate, ::decimate]
    shape = f_d.shape
    gnx, gny, gnz = g_d.shape

    c = xp.array([(n - 1) / 2.0 for n in shape], dtype=xp.float64)
    ix, iy, iz = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]),
                              xp.arange(shape[2]), indexing='ij')
    Xf = xp.stack([xp.ravel(ix), xp.ravel(iy), xp.ravel(iz)], axis=0).astype(xp.float64)
    c_col = c.reshape(3, 1)

    f_flat = xp.ravel(f_d)

    if t0 is None:
        t0_dec = fft_rigid_shift(f_d, g_d, xp)
    else:
        t0_dec = np.asarray(t0, dtype=np.float64) / decimate

    def warp_and_cost(p):
        A = _build_affine_matrix(p)
        t = p[6:9]
        A_x = xp.asarray(A, dtype=xp.float64)
        t_x = xp.asarray(t, dtype=xp.float64).reshape(3, 1)
        Xp = A_x @ (Xf - c_col) + c_col + t_x
        in_bounds = ((Xp[0] >= 0) & (Xp[0] <= gnx - 1) &
                     (Xp[1] >= 0) & (Xp[1] <= gny - 1) &
                     (Xp[2] >= 0) & (Xp[2] <= gnz - 1))
        g_samp = _map_coordinates(g_d, Xp, xp, order=1)
        # Correlate over the full in-bounds field (pore/zero voxels kept as
        # legitimate low-intensity contrast): the material/pore boundary IS
        # the registration signal here -- see this function's docstring.
        return _zncc_masked(f_flat, g_samp, in_bounds, xp)

    p_identity = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])
    p_translation_only = p_identity.copy()
    p_translation_only[6:9] = t0_dec

    cost_identity = warp_and_cost(p_identity)
    cost_translation_only = warp_and_cost(p_translation_only)

    bounds = ([(-angle_bound, angle_bound)] * 3
              + [(1 - scale_bound, 1 + scale_bound)] * 3
              + [(None, None)] * 3)

    result = minimize(warp_and_cost, p_translation_only, method='Powell',
                       bounds=bounds, options={'maxiter': maxiter, 'disp': disp})

    p_opt = result.x
    A = _build_affine_matrix(p_opt)
    t_dec = p_opt[6:9]
    c_dec = np.asarray(c.get() if hasattr(c, 'get') else c)

    t_full = t_dec * decimate
    c_full = c_dec * decimate

    info = {
        'cost_identity': cost_identity,
        'cost_translation_only': cost_translation_only,
        'cost_final': float(result.fun),
        'nfev': result.nfev,
        'p': p_opt,
    }
    return A, t_full, c_full, info


def _gaussian_filter(xp):
    if xp is np:
        from scipy.ndimage import gaussian_filter
    else:
        from cupyx.scipy.ndimage import gaussian_filter
    return gaussian_filter


def _smooth_plain(img, sigma, xp):
    """Gaussian-smooth a fully-valid image (no masking)."""
    if sigma <= 0:
        return img
    return _gaussian_filter(xp)(img, sigma, mode="nearest")


def _smooth_masked(img, valid_f, sigma, xp):
    """Normalized-convolution Gaussian smooth: smooth(img*valid)/smooth(valid),
    so the background fill never bleeds across the material/void boundary. The
    validity mask `valid_f` (float) always comes from the ORIGINAL image."""
    if sigma <= 0:
        return img
    gf = _gaussian_filter(xp)
    num = gf(img * valid_f, sigma, mode="constant", cval=0.0)
    den = gf(valid_f, sigma, mode="constant", cval=0.0)
    out = img.copy()
    good = den > 1e-6
    out[good] = num[good] / den[good]
    return out


def affine_refine_gn(f_pix, g_pix, xp, A0, t0, c, sigmas=(4.0, 2.0, 1.0, 0.0),
                     max_iter=50, tol=1e-6, patience=3, cval=0.0,
                     chunk_voxels=8_000_000, disp=False):
    """Refine a full 3x3 affine (rotation + scale + shear) + translation by
    forward-additive Gauss-Newton (Lucas-Kanade), starting from the coarse
    (A0, t0) produced by affine_prealign.

    Convention. The returned (A, t) are in the affine_prealign /
    warp_onto_ref_grid convention -- the warp samples g at x' = A (x - c) + c
    + t and compares to f(x) -- so they plug straight into the existing
    downstream (ext_affine, warp_onto_ref_grid, the deformed-bbox corners)
    with no sign changes.

    Internally, though, the Gauss-Newton solve runs in the INVERSE
    (generative) convention: it deforms f=ref by u(x) = M (x - c) + t_r,
    samples ref, and compares to g=def -- i.e. it fits def(x) ~= ref(x -
    u(x)). This is deliberate and is the reason the refinement is stable and
    accurate where affine_prealign is not: the bundled ground-truth def was
    itself generated by exactly this backward (pull) warp of ref, so
    deforming ref reproduces the identical interpolation and boundary/fill
    behaviour and the residual can drive to near zero. Warping the OTHER way
    (deforming def, as the coarse ZNCC search and the downstream warp do)
    inverts the fill boundary and makes full-resolution Gauss-Newton chase
    the material/void step edge and diverge. See recover_affine.py's
    docstring for the same argument. The result is converted back at the end
    by A = (I - M)^-1, t = A @ t_r (exact for an affine about a fixed c).

    Why this exists
    ---------------
    affine_prealign uses a derivative-free Powell search over a 9-DOF
    (rotation + per-axis scale, NO shear) model on a *decimated* volume,
    maximizing ZNCC. That is robust for large rigid offsets but coarse: on
    the elastic ground-truth pair it misses the dominant ~5% z-compression
    entirely (recovers s_zz=1.003 vs. the true ~0.95) because a few-percent
    normal strain is sub-voxel once the volume is decimated by 4, and Powell
    has no gradient to follow it. This refinement fixes exactly that:

      * analytic Gauss-Newton with the exact 12-parameter affine Jacobian
        (dW/dM_ij = -grad_i(ref)|_s * (x - c)_j, dW/dt_i = -grad_i(ref)|_s),
        solved in float64 -- follows the sub-voxel strain gradient directly;
      * FULL resolution (no decimation) for sub-voxel precision;
      * a full 3x3 M, so shear is representable;
      * a Gaussian SCALE-SPACE pyramid (large sigma -> 0) to widen the
        capture basin, carrying (M, t_r) between levels. Smoothing does not
        change the coordinate grid, so M/t_r need no rescaling between levels.

    Memory is bounded by assembling the 12x12 normal equations over axis-0
    slabs (`chunk_voxels` voxels per slab), so the full-resolution solve never
    materializes a full (12, Nvox) Jacobian and scales to large CT volumes.

    Parameters
    ----------
    f_pix, g_pix : array (Nx, Ny, Nz)
        Reference / deformed volumes (numpy or cupy, matching xp). Same shape.
    A0 : (3,3) array, t0 : (3,) array, c : (3,) array
        Initial affine, translation, and (fixed) center from affine_prealign,
        in full-resolution voxel units (prealign convention).
    sigmas : sequence of float
        Gaussian smoothing levels, coarse first; end with 0.0 for a
        full-resolution final refinement.
    max_iter, tol : per-level iteration cap and ||dp|| convergence tolerance.
    patience : int
        Stop a level early after this many iterations with no new minimum in
        ||dp|| -- avoids burning the full max_iter spinning on the GPU-noise
        floor (~1e-5, above tol) once the fit has effectively converged.
    cval : float
        Fill value of both volumes' background (masked out of the fit; also
        used as the out-of-bounds sample value).
    chunk_voxels : int
        Approximate voxels per axis-0 slab in the normal-equation assembly.

    Returns
    -------
    A : (3,3) ndarray, t : (3,) ndarray  (prealign convention, full-res voxels)
    info : dict with 'rms_initial', 'rms_final', 'iters' (list, per level).
    """
    f = xp.asarray(f_pix, dtype=xp.float32)
    g = xp.asarray(g_pix, dtype=xp.float32)
    shape = f.shape
    Nx, Ny, Nz = shape
    Nrc = Ny * Nz

    I3 = np.eye(3)
    A0 = np.asarray(A0, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64)
    # prealign (A, t) -> generative (M, t_r):  A = (I-M)^-1, t = A t_r.
    A0inv = np.linalg.inv(A0)
    M0 = I3 - A0inv
    tr0 = A0inv @ t0

    c = np.asarray(c, dtype=np.float64)
    c_x = xp.asarray(c, dtype=xp.float32)
    Nbound = xp.asarray([n - 1 for n in shape], dtype=xp.float32)

    valid_g = (g != cval)                                # def validity (fit mask)

    M = xp.asarray(M0, dtype=xp.float64)
    tr = xp.asarray(tr0, dtype=xp.float64)

    rows_per_chunk = max(1, int(chunk_voxels // max(Nrc, 1)))

    # 1-D coordinate arms (centred), reused across slabs / iterations.
    Ycen = (xp.arange(Ny, dtype=xp.float32) - c_x[1])[:, None]   # (Ny, 1)
    Zcen = (xp.arange(Nz, dtype=xp.float32) - c_x[2])[None, :]   # (1, Nz)

    def _slab_coords(Mf, trf, i0, i1):
        """Sample locations s(x) = x - M(x-c) - t_r for ref-grid voxels in the
        axis-0 slab [i0, i1). Returns (Xc list of 3, coords (3,ns,Ny,Nz))."""
        ns = i1 - i0
        Xcen = (xp.arange(i0, i1, dtype=xp.float32) - c_x[0])[:, None, None]
        Xc = [xp.broadcast_to(Xcen, (ns, Ny, Nz)),
              xp.broadcast_to(Ycen[None], (ns, Ny, Nz)),
              xp.broadcast_to(Zcen[None], (ns, Ny, Nz))]
        gx = xp.arange(i0, i1, dtype=xp.float32)[:, None, None]
        base = [xp.broadcast_to(gx, (ns, Ny, Nz)),
                xp.broadcast_to(xp.arange(Ny, dtype=xp.float32)[None, :, None],
                                (ns, Ny, Nz)),
                xp.broadcast_to(xp.arange(Nz, dtype=xp.float32)[None, None, :],
                                (ns, Ny, Nz))]
        coords = xp.stack([
            base[i] - (Mf[i, 0] * Xc[0] + Mf[i, 1] * Xc[1] + Mf[i, 2] * Xc[2])
            - trf[i] for i in range(3)], axis=0)
        return Xc, coords

    def _rms(M_, tr_):
        """Masked RMS intensity residual e = def - ref(x - u), over slabs."""
        Mf = M_.astype(xp.float32)
        trf = tr_.astype(xp.float32)
        se = 0.0
        n = 0.0
        for i0 in range(0, Nx, rows_per_chunk):
            i1 = min(i0 + rows_per_chunk, Nx)
            _, coords = _slab_coords(Mf, trf, i0, i1)
            warped = _map_coordinates(f, coords, xp, order=1, mode="constant",
                                      cval=cval)
            inb = xp.all((coords >= 0) & (coords <= Nbound[:, None, None, None]),
                         axis=0)
            m = inb & valid_g[i0:i1]
            e = (g[i0:i1] - warped)[m]
            se += float((e * e).sum(dtype=xp.float64))
            n += float(m.sum())
        return float(np.sqrt(se / max(n, 1.0)))

    rms_initial = _rms(M, tr)
    iters = []

    for sigma in sigmas:
        f_s = _smooth_plain(f, sigma, xp)
        # Normalized-convolution smooth of def keeps its fill from bleeding.
        g_s = _smooth_masked(g, valid_g.astype(xp.float32), sigma, xp)
        # Gradient of the (smoothed) reference, sampled at s(x) below.
        gg = [xp.ascontiguousarray(comp) for comp in xp.gradient(f_s)]

        n_it = 0
        best_step = np.inf
        stall = 0
        for it in range(max_iter):
            Mf = M.astype(xp.float32)
            trf = tr.astype(xp.float32)
            H = xp.zeros((12, 12), dtype=xp.float64)
            b = xp.zeros(12, dtype=xp.float64)

            for i0 in range(0, Nx, rows_per_chunk):
                i1 = min(i0 + rows_per_chunk, Nx)
                Xc, coords = _slab_coords(Mf, trf, i0, i1)

                warped = _map_coordinates(f_s, coords, xp, order=1,
                                          mode="constant", cval=cval)
                inb = xp.all((coords >= 0) &
                             (coords <= Nbound[:, None, None, None]), axis=0)
                mask = (inb & valid_g[i0:i1]).astype(xp.float32)
                e = (g_s[i0:i1] - warped) * mask

                # grad_i(ref) at the sample locations s = coords.
                gs = [_map_coordinates(gg[i], coords, xp, order=1,
                                       mode="nearest") for i in range(3)]

                # 12 Jacobian channels J_k = dW/dp_k, W = ref(x - u), params
                #   M00 M01 M02 M10 M11 M12 M20 M21 M22  t0 t1 t2.
                J = []
                for i in range(3):
                    for j in range(3):
                        J.append(-gs[i] * Xc[j] * mask)    # dW/dM_ij
                for i in range(3):
                    J.append(-gs[i] * mask)                # dW/dt_i

                # Accumulate the normal equations in float64 (as recover_affine
                # does): a float32 dot product over millions of voxels per slab
                # loses precision in H/b.
                Jf = xp.stack([Jk.ravel() for Jk in J], axis=0).astype(xp.float64)
                H += Jf @ Jf.T
                b += Jf @ e.ravel().astype(xp.float64)

            # Levenberg damping guards a degenerate channel.
            H += xp.eye(12, dtype=xp.float64) * (
                1e-9 * float(xp.trace(H)) / 12.0 + 1e-30)
            dp = xp.linalg.solve(H, b)
            M = M + dp[:9].reshape(3, 3)
            tr = tr + dp[9:]

            step = float(xp.linalg.norm(dp))
            n_it = it + 1
            if disp:
                print(f"    sigma={sigma:g} iter {it:2d}: ||dp||={step:.3e}",
                      flush=True)
            if step < tol:
                break
            # Patience stop: bail once ||dp|| stops finding new minima (it has
            # settled on the GPU-noise floor, which sits above tol).
            if step < best_step - 1e-12:
                best_step = step
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break
        iters.append(n_it)

    rms = _rms(M, tr)
    M_np = np.asarray(M.get() if hasattr(M, 'get') else M)
    tr_np = np.asarray(tr.get() if hasattr(tr, 'get') else tr)
    # generative (M, t_r) -> prealign (A, t):  A = (I-M)^-1, t = A t_r.
    A_np = np.linalg.inv(I3 - M_np)
    t_np = A_np @ tr_np
    info = {'rms_initial': rms_initial, 'rms_final': rms, 'iters': iters}
    return A_np, t_np, info


def correlate_gpu(f_pix, g_pix, Nx, Ny, Nz, h, xp, g_origin=(0, 0, 0),
                   glt=0.0,
                   l0=None, K_ref_laplacian=None, reg_exponent=2, c_reg=1.0,
                   reg_type='laplacian', K_ref_elastic=None, eps_reg=1e-3,
                   dynamic_mask=False, freeze_mask_after=None, U0=None,
                   maxiter=30, eps=1e-3, pcg_tol=1e-6, pcg_max_iter=2000,
                   disp=True, dtype=np.float32):
    """Matrix-free GPU DVC correlation loop, reproducing dic.Correlate for
    a voxel-center-quadrature mesh covering the full reference image.

    Parameters
    ----------
    f_pix, g_pix : array, shape (Nx*h+1, Ny*h+1, Nz*h+1)
        Reference / deformed image voxel data (numpy or cupy, matching xp).
    Nx, Ny, Nz, h : int
        Mesh element counts and voxel-center quadrature edge length.
    xp : numpy or cupy module
    g_origin : (x0, y0, z0)
        Origin of g_pix in the same coordinate system as f_pix (voxel
        units). Used for the deformed-image bounds check / sampling.
    glt : float, default 0.0
        Gray-level threshold: voxels with intensity <= glt (in the reference
        f, via mask0_vox = f_vox > glt, or in the sampled deformed g) are
        treated as "no material / no contrast" and excluded from the mask,
        H/b assembly, and normalization. 0.0 reproduces the historical
        "nonzero-intensity" convention.
    l0 : float, optional
        If given, enables regularization with weight `ll = c_reg * (l0/T)**
        reg_exponent * H0/L0`, T = 10*h.
    reg_type : 'laplacian' or 'equilibrium_gap'
        'laplacian' (default): R = K_ref_laplacian (Tikhonov, 2nd order,
        reg_exponent=2, default). H0/L0 = V.H.V / V.R.V for a plane wave
        V, mirroring dic.Correlate's calibration.
        'equilibrium_gap': R = R_m + eps_reg * K_ref_laplacian, with
        R_m = K_i^T K_i = K @ P_i @ K the elastic-stiffness-based
        equilibrium-gap operator (mesh.EquilibriumGapRegularizer,
        kernels_dvc.matvec_equilibrium_gap; 4th order, reg_exponent=4 is
        the matching exponent). H0/L0 = trace(H)/trace(R) (guide eq. 5.5)
        instead of the plane-wave ratio -- per dvc_next_steps.md item 1,
        the plane-wave V.H.V/V.R.V ratio under-estimates ll for this
        4th-order operator by ~4 orders of magnitude, while
        trace(H)/trace(R) gives a well-conditioned ll across c_reg in
        [1e-4, 1e2] (strain_xx_fit ~0.0099 vs. the imposed 0.01 on the
        Nx=8 tension benchmark, for any c_reg in that range); c_reg=1.0
        (default) is a reasonable choice.
    K_ref_laplacian : (24, 24) array, optional
        Required if l0 is given (geometry_dvc.build_K_ref_laplacian()).
    K_ref_elastic : (24, 24) array, optional
        Required if l0 is given and reg_type='equilibrium_gap'
        (geometry_dvc.build_K_ref(nu)).
    dynamic_mask : bool
        If True, H is rebuilt every GN iteration from the dynamic mask
        (mask0 & g.InBounds(U)), following dvc_next_steps.md item 2 / plan
        Section 6 -- partial-overlap voxels stop contributing to H as well
        as to b. If False (default), H uses the static mask0 only
        (frozen-H, matches dic.Correlate's CPU behaviour and the benchmark
        numbers in CLAUDE.md).

        Caveat: at a tension boundary, voxels can flip in/out of the
        dynamic mask from one GN iteration to the next as `dU` fluctuates
        near the in-bounds threshold, which can prevent `dU/U` from
        decreasing below `eps` (it plateaus at the flip amplitude instead)
        even though the displacement field itself is converged. Use
        dynamic_mask=True for genuinely partial-overlap problems (where a
        static mask0 would be wrong), but check convergence behaviour
        case-by-case. See also freeze_mask_after.
    freeze_mask_after : int or None, default None
        If not None, after iteration `freeze_mask_after` (0-indexed) the
        per-voxel in-bounds mask computed at that iteration is frozen and
        reused for all subsequent b (and, when dynamic_mask=True, also H)
        computations. This eliminates the per-iteration mask-flip 2-cycle
        that occurs when a small number of boundary voxels flip in/out of
        bounds as dU oscillates near the in-bounds threshold, preventing
        dU/U from dropping below eps even though the displacement field is
        effectively converged. Typical value: freeze_mask_after=2 (freeze
        after the second GN step, once the warm-start error has been
        corrected). Has no effect when all voxels are always in-bounds.
    U0 : (Ndof,) array, optional
        Initial displacement DOF vector, in this call's voxel units (i.e.
        already scaled for `h`). Defaults to zeros. Used by
        multiscale_correlate_gpu to warm-start finer levels from coarser
        ones.
    dtype : numpy dtype, default np.float32
        Floating-point dtype for all per-voxel/per-DOF field arrays
        (f_pix_x, g_pix_x, grad_f_vox, U, res, etc.). float32 roughly
        halves VRAM use for the dominant Nvox-sized arrays; pass
        np.float64 for exact-reproducibility comparisons.

    Returns
    -------
    U : (Ndof,) array
    res : (Nvox,) array -- final residual at every voxel.
    dU : (Ndof,) array -- the Gauss-Newton update applied at the *last*
        iteration (the increment that produced the returned U). Its per-node
        magnitude relative to U is the local analog of the printed global
        convergence measure dU/U, so write_outputs turns it into a per-cell
        convergence field. All-zeros if maxiter < 1.
    """
    Nvx, Nvy, Nvz = Nx * h + 1, Ny * h + 1, Nz * h + 1
    Ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
    Nvox = Nvx * Nvy * Nvz

    # Element/voxel/DOF indices are kept in int32 (chunked dofs/vox_ids
    # tensors are otherwise a significant fraction of VRAM); this requires
    # Nvox and Ndof to fit in int32.
    if Nvox > np.iinfo(np.int32).max or Ndof > np.iinfo(np.int32).max:
        raise ValueError(
            "Nvox=%d / Ndof=%d exceed int32 range; matrix-free index "
            "arrays assume int32 indices" % (Nvox, Ndof))

    f_pix_x = xp.asarray(f_pix, dtype=dtype)
    g_pix_x = xp.asarray(g_pix, dtype=dtype)

    f_vox, grad_f_vox = image_grad_and_values(f_pix_x, xp, dtype=dtype)
    del f_pix_x

    # Reference-side material mask: zero intensity means "no material / no
    # contrast" (see CLAUDE.md), so these voxels carry no correlation signal
    # and are excluded from H/b assembly and normalization stats below.
    # Additionally, zero grad_f_vox for voxels touching the material/air
    # interface (6-connectivity, matching xp.gradient's central-difference
    # stencil) -- otherwise the intensity step at the interface produces a
    # large spurious gradient that is not real material texture.
    mask0_vox = f_vox > glt
    grad_mask = erode_mask0_vox(mask0_vox, Nvx, Nvy, Nvz, xp)
    grad_f_vox[~grad_mask] = 0
    del grad_mask  # only used to zero grad_f_vox at the interface

    # mean()/std() normalization is taken over the quadrature-point array
    # (with shared-voxel multiplicity), matching dic.ComputeLHS/ComputeRHS,
    # restricted to mask0_vox (excluding no-material voxels). Computed by a
    # chunked reduction (masked_quadpoint_mean_std) rather than gathering the
    # full Ne*(h+1)^3 quad-point array.
    mean0, std0 = masked_quadpoint_mean_std(f_vox, mask0_vox, h, Nx, Ny, Nz, xp)
    f_vox -= mean0

    N_stencil, w_stencil = build_N_stencil(h)

    if reg_type not in ('laplacian', 'equilibrium_gap'):
        raise ValueError("reg_type must be 'laplacian' or 'equilibrium_gap'")

    interior_mask = None
    if reg_type == 'equilibrium_gap':
        from .geometry_dvc import build_interior_dof_mask
        interior_mask = xp.asarray(build_interior_dof_mask(Nx, Ny, Nz), dtype=dtype)

    def matvec_reg(v):
        if reg_type == 'laplacian':
            return matvec_K_ref(v, K_ref_laplacian, E_elem=h, Nx=Nx, Ny=Ny, Nz=Nz, xp=xp)
        Rm = matvec_equilibrium_gap(v, K_ref_elastic, interior_mask, E_elem=h,
                                      Nx=Nx, Ny=Ny, Nz=Nz, xp=xp)
        Rt = matvec_K_ref(v, K_ref_laplacian, E_elem=h, Nx=Nx, Ny=Ny, Nz=Nz, xp=xp)
        return Rm + eps_reg * Rt

    def diag_reg():
        if reg_type == 'laplacian':
            return build_diagonal_K_ref(K_ref_laplacian, h, Nx, Ny, Nz, Ndof, xp)
        diag_m = build_diagonal_equilibrium_gap(K_ref_elastic, interior_mask, h,
                                                  Nx, Ny, Nz, Ndof, xp)
        diag_t = build_diagonal_K_ref(K_ref_laplacian, h, Nx, Ny, Nz, Ndof, xp)
        return diag_m + eps_reg * diag_t

    ll = 0.0
    if l0 is not None:
        if K_ref_laplacian is None:
            raise ValueError("K_ref_laplacian is required when l0 is given")
        if reg_type == 'equilibrium_gap' and K_ref_elastic is None:
            raise ValueError("K_ref_elastic is required when l0 is given and "
                              "reg_type='equilibrium_gap'")
        T = 10 * h
        if reg_type == 'equilibrium_gap':
            # Trace-based calibration (guide eq. 5.5): ll = c_reg*(l0/T)**
            # reg_exponent*trace(H)/trace(R). The plane-wave H0=V.H.V,
            # L0=V.R.V ratio below is poorly conditioned for the 4th-order
            # R_m operator (dvc_next_steps.md item 1: under-estimates ll by
            # ~4 orders of magnitude); trace(H)/trace(R) is the CPU-side
            # fix (test_dvc_elastic_reg.py) and is cheap here since
            # build_diagonal_H/diag_reg are already needed for the Jacobi
            # preconditioner.
            diag_H_cal = build_diagonal_H(grad_f_vox, mask0_vox, N_stencil, w_stencil,
                                           h, Nx, Ny, Nz, Ndof, xp)
            H0 = float(diag_H_cal.sum())
            L0 = float(diag_reg().sum())
        else:
            # Plane-wave calibration (mirrors dic.Correlate): build a plane
            # wave V over the mesh nodes, ll = c_reg*(l0/T)**reg_exponent*
            # H0/L0 with H0 = V.H.V, L0 = V.R.V, both via matrix-free
            # matvecs.
            node_idx = xp.arange((Nx + 1) * (Ny + 1) * (Nz + 1))
            jj = (node_idx // (Nx + 1)) % (Ny + 1)
            V_scalar = xp.cos(jj.astype(dtype) * h / T * 2 * np.pi)
            V = xp.zeros(Ndof, dtype=dtype)
            V[0::3] = V_scalar  # plane wave on the x-displacement component

            Hv = matvec_H(V, grad_f_vox, mask0_vox, N_stencil, w_stencil, h, Nx, Ny, Nz, xp)
            Lv = matvec_reg(V)
            H0 = float(xp.dot(V, Hv))
            L0 = float(xp.dot(V, Lv))
        ll = c_reg * (l0 / T) ** reg_exponent * H0 / L0
        if disp:
            print('Regularization: %s with lambda = %2.3e' % (reg_type, ll))
    else:
        if disp:
            print('Regularization: None')


    if U0 is None:
        U = xp.zeros(Ndof, dtype=dtype)
    else:
        U = xp.asarray(U0, dtype=dtype).copy()
    res = xp.zeros(Nvox, dtype=dtype)
    dU = xp.zeros(Ndof, dtype=dtype)  # last GN update; overwritten in the loop

    # Frozen-H setup (dynamic_mask=False): build once outside the loop.
    if not dynamic_mask:
        diag = build_diagonal_H(grad_f_vox, mask0_vox, N_stencil, w_stencil, h, Nx, Ny, Nz, Ndof, xp)
        if l0 is not None:
            diag = diag + ll * diag_reg()
        eps_zero = 1e-5 * float(diag.min())
        M_inv = 1.0 / (diag + eps_zero)

        def matvec_full(v, mask_H=mask0_vox):
            out = matvec_H(v, grad_f_vox, mask_H, N_stencil, w_stencil, h, Nx, Ny, Nz, xp)
            if l0 is not None:
                out = out + ll * matvec_reg(v)
            return out

    b_mask_frozen = None  # set once freeze_mask_after is reached

    # Return the setup-phase transients (xp.gradient outputs, F-order ravels,
    # erosion/normalization temporaries, etc.) to the driver before the GN
    # loop. They are already dead (out of scope / del'd), but the CuPy pool
    # would otherwise keep those blocks reserved for the whole run, inflating
    # the process's resident VRAM well above the loop's working set.
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()

    for ik in range(maxiter):
        # Sample g at every voxel's deformed position x+u(x) and the per-voxel
        # active mask (in_bounds & mask0 & g>0) in element chunks, so the full
        # (3, Nvox) deformed-coordinate array is never materialized -- it is the
        # largest per-iteration buffer and sets the sampling-stage VRAM peak.
        # Bit-identical to the un-chunked interp + single map_coordinates call
        # (see sample_deformed_field). mask_vox already folds in the g>0 check.
        g_vals, mask_vox = sample_deformed_field(
            U, g_pix_x, N_stencil, h, Nx, Ny, Nz, xp,
            g_origin=g_origin, mask0_vox=mask0_vox, glt=glt)

        # Freeze the in-bounds mask for b (and H when dynamic_mask=True) once
        # requested, to break the 2-cycle caused by boundary voxels flipping
        # in/out of bounds as dU oscillates near the in-bounds threshold.
        if freeze_mask_after is not None and b_mask_frozen is None and ik >= freeze_mask_after:
            b_mask_frozen = mask_vox.copy()
        b_mask = b_mask_frozen if b_mask_frozen is not None else mask_vox

        # ZNCC normalization stats over the quad-point array, restricted to
        # b_mask (not the live mask_vox) -- using b_mask eliminates the
        # normalization-induced oscillation that persists even after b_mask is
        # frozen, since g_mean/std1 from a flipping mask_vox produce a changing
        # residual even at a fixed displacement. Both stats come from ONE
        # chunked pass over the raw g_vals: std is shift-invariant, so
        # std(g_vals - g_mean) == std(g_vals), and masked_quadpoint_mean_std
        # never builds the full Ne*(h+1)^3 gather array.
        g_mean, std1 = masked_quadpoint_mean_std(g_vals, b_mask, h, Nx, Ny, Nz, xp)
        g_vals -= g_mean

        # Reuse g_vals's buffer for res: res = f_vox - (std0/std1)*g_vals.
        g_vals *= -(std0 / std1)
        g_vals += f_vox
        res = g_vals
        # Zero the residual outside b_mask -- the mask compute_b_gpu/matvec_H
        # actually sum over.  Before freeze_mask_after, b_mask IS the live
        # mask_vox, so this is unchanged.  After freeze, zeroing by b_mask
        # (not the live mask_vox) is essential: a boundary voxel that is in
        # the frozen b_mask but flips out of the live mask_vox must keep its
        # (boundary-clamped) residual rather than being zeroed -- otherwise
        # res, and hence b, keeps changing as such voxels flip in/out of
        # mask_vox, which is the residual-side source of the dU/U 2-cycle
        # plateau that freezing b_mask alone does not remove.
        res[~b_mask] = 0.0

        b = compute_b_gpu(res, b_mask, grad_f_vox, N_stencil, w_stencil, h, Nx, Ny, Nz, Ndof, xp)
        if l0 is not None:
            b = b - ll * matvec_reg(U)

        if dynamic_mask:
            # Rebuild H (and its Jacobi preconditioner) from b_mask (frozen
            # once freeze_mask_after is reached, current mask_vox otherwise).
            diag = build_diagonal_H(grad_f_vox, b_mask, N_stencil, w_stencil, h, Nx, Ny, Nz, Ndof, xp)
            if l0 is not None:
                diag = diag + ll * diag_reg()
            eps_zero = 1e-5 * float(diag.min())
            M_inv = 1.0 / (diag + eps_zero)

            def matvec_full(v, mask_H=b_mask):
                out = matvec_H(v, grad_f_vox, mask_H, N_stencil, w_stencil, h, Nx, Ny, Nz, xp)
                if l0 is not None:
                    out = out + ll * matvec_reg(v)
                return out

        dU = pcg(matvec_full, b, M_inv, xp, max_iter=pcg_max_iter, tol=pcg_tol)
        U = U + dU

        err = float(xp.linalg.norm(dU) / xp.linalg.norm(U))
        if disp:
            _, stdr = masked_quadpoint_mean_std(res, mask_vox, h, Nx, Ny, Nz, xp)
            n_active = int(mask_vox.sum())
            print("Iter # %2d | std(res)=%2.2f gl | dU/U=%1.2e | active pts=%d/%d (%.1f%%)"
                  % (ik + 1, stdr, err, n_active, Nvox, 100 * n_active / Nvox))
        if err < eps:
            break

    return U, res, dU


def multiscale_correlate_gpu(f_pix, g_pix, Nx, Ny, Nz, h, xp, scales=(2, 1, 0),
                              g_origin=(0, 0, 0), l0=None, fft_prealign=True,
                              prealign_affine=False, ext_affine=None,
                              **kwargs):
    """Coarse-to-fine warm start for correlate_gpu (plan Section 8).

    Runs correlate_gpu repeatedly on strided-decimated copies of f_pix/g_pix,
    coarsest scale first, carrying the displacement solution forward as the
    next (finer) level's U0. The element grid (Nx, Ny, Nz) is fixed across
    levels -- only the voxel-center quadrature density `h` changes, so Ndof
    is identical at every level and U0 can be passed through directly after
    rescaling for the change in voxel size.

    Parameters
    ----------
    f_pix, g_pix : array, shape (Nx*h+1, Ny*h+1, Nz*h+1)
        Reference / deformed image voxel data, at the finest (scale 0)
        resolution.
    Nx, Ny, Nz, h : int
        Element grid and voxel-center quadrature edge length at the finest
        resolution. `h` must be divisible by 2**iscale for every `iscale`
        in `scales`.
    scales : sequence of int, default (2, 1, 0)
        Pyramid levels, coarsest first (any order is accepted -- sorted
        descending internally). Level `iscale` subsamples f_pix/g_pix by a
        stride of `2**iscale` (so `h_i = h // 2**iscale`); scale 0 is the
        full-resolution image.
    g_origin : (x0, y0, z0)
        Origin of g_pix at the finest resolution. Must be divisible by
        2**iscale for every requested scale.
    fft_prealign : bool, default True
        If True (and prealign_affine is False), before the pyramid loop,
        estimate a rigid voxel translation between the coarsest-scale f/g
        images via FFT phase correlation (fft_rigid_shift) and seed the
        coarsest level's U0 with this uniform displacement
        (dvc_next_steps.md item 4). Helps when the true displacement is
        large relative to the coarsest element size, which a zero initial
        guess cannot reach.
    prealign_affine : bool, default False
        If True, takes precedence over fft_prealign: before the pyramid
        loop, estimate a 9-DOF affine alignment (rotation + independent
        per-axis scale + translation, see affine_prealign) between the
        coarsest-scale f/g images and seed the coarsest level's U0 with the
        corresponding uniform-affine displacement field
        u(x) = A @ (x - c) + c + t - x, evaluated at each node's
        reference voxel coordinate (i*h_i, j*h_i, k*h_i).
    ext_affine : tuple (A, t, c) or None, default None
        Pre-computed affine parameters to use instead of running
        affine_prealign. Takes precedence over both prealign_affine and
        fft_prealign. A is the (3,3) affine matrix, t the (3,) translation
        and c the (3,) center of rotation, all in full-resolution voxel
        units of f_pix. Useful when affine parameters were pre-computed
        externally (e.g. by preprocess.py) and need to be adjusted
        for a coordinate offset between f_pix and the original image (see
        run_dvc.py: c_adj = c - crop_lo, t_adj = t + crop_lo).
    l0 : float, optional
        Regularization length at the finest resolution (voxel units, as in
        correlate_gpu). Internally rescaled to `l0 / 2**iscale` at each
        coarser level, so the calibration ll = c_reg*(l0_i/T_i)**reg_exponent
        with T_i = 10*h_i stays consistent across levels.
    **kwargs
        Forwarded to correlate_gpu (glt, K_ref_laplacian, reg_exponent, c_reg,
        reg_type, K_ref_elastic, eps_reg, dynamic_mask, maxiter, eps,
        pcg_tol, pcg_max_iter, disp).

    Returns
    -------
    U : (Ndof,) array
    res : (Nvox,) array -- final residual at every voxel, at the finest
        resolution.
    dU : (Ndof,) array -- the last Gauss-Newton update of the finest-scale
        solve (finest-resolution voxel units), forwarded from correlate_gpu
        for the per-cell convergence field (see correlate_gpu Returns).
    """
    scales = sorted(set(scales), reverse=True)
    dtype = kwargs.get('dtype', np.float32)

    A_aff = t_aff = c_aff = None
    A_aff = t_aff = c_aff = None
    if ext_affine is not None:
        A_aff, t_aff, c_aff = ext_affine
        A_aff = np.asarray(A_aff, dtype=np.float64)
        t_aff = np.asarray(t_aff, dtype=np.float64)
        c_aff = np.asarray(c_aff, dtype=np.float64)
        if kwargs.get('disp', True):
            print("External affine: t = (%.1f, %.1f, %.1f) voxels, "
                  "c = (%.1f, %.1f, %.1f)" % (*t_aff, *c_aff))
    elif prealign_affine:
        A_aff, t_aff, c_aff, aff_info = affine_prealign(f_pix, g_pix, xp)
        if kwargs.get('disp', True):
            rx, ry, rz = (180 / np.pi) * np.array([
                np.arctan2(A_aff[2, 1], A_aff[2, 2]),
                np.arctan2(-A_aff[2, 0], np.hypot(A_aff[2, 1], A_aff[2, 2])),
                np.arctan2(A_aff[1, 0], A_aff[0, 0])])
            sx, sy, sz = np.linalg.norm(A_aff, axis=0)
            print("Affine pre-alignment (full res): rotation (deg) = (%.3f, %.3f, %.3f), "
                  "scale = (%.4f, %.4f, %.4f), t = (%.1f, %.1f, %.1f) voxels "
                  "(1-ZNCC %.4f -> %.4f)"
                  % (rx, ry, rz, sx, sy, sz, t_aff[0], t_aff[1], t_aff[2],
                     aff_info['cost_translation_only'], aff_info['cost_final']))

    U = None
    prev_s = None
    res = None
    dU = None
    for iscale in scales:
        s = 2 ** iscale
        if h % s != 0:
            raise ValueError("h=%d not divisible by 2**%d=%d" % (h, iscale, s))
        if any(o % s != 0 for o in g_origin):
            raise ValueError("g_origin=%r not divisible by 2**%d=%d for all "
                              "requested scales" % (g_origin, iscale, s))

        h_i = h // s
        f_i = xp.asarray(f_pix)[::s, ::s, ::s]
        g_i = xp.asarray(g_pix)[::s, ::s, ::s]
        g_origin_i = tuple(o // s for o in g_origin)
        l0_i = None if l0 is None else l0 / s

        if U is not None:
            U = U * (prev_s / s)  # rescale displacement to this level's voxel units
        elif A_aff is not None:
            # Rescale the full-resolution affine (A, t, c) to this level's
            # voxel units: X_full = s * X_i, and A is scale-invariant under
            # uniform coordinate rescaling, so c_i = c/s, t_i = t/s give
            # u_i(X_i) = A @ (X_i - c_i) + c_i + t_i - X_i = u_full(s*X_i)/s.
            c_i = c_aff / s
            t_i = t_aff / s
            n_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
            node_idx = np.arange(n_nodes)
            ni = node_idx % (Nx + 1)
            nj = (node_idx // (Nx + 1)) % (Ny + 1)
            nk = node_idx // ((Nx + 1) * (Ny + 1))
            X_nodes = np.stack([ni, nj, nk], axis=1).astype(np.float64) * h_i
            u_nodes = (X_nodes - c_i) @ A_aff.T + c_i + t_i - X_nodes
            U = xp.zeros(3 * n_nodes, dtype=dtype)
            U[0::3] = xp.asarray(u_nodes[:, 0], dtype=dtype)
            U[1::3] = xp.asarray(u_nodes[:, 1], dtype=dtype)
            U[2::3] = xp.asarray(u_nodes[:, 2], dtype=dtype)
        elif fft_prealign:
            d = fft_rigid_shift(f_i, g_i, xp)
            n_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
            U = xp.zeros(3 * n_nodes, dtype=dtype)
            U[0::3] = d[0]
            U[1::3] = d[1]
            U[2::3] = d[2]
            if kwargs.get('disp', True):
                print("FFT pre-alignment: rigid shift d = (%.1f, %.1f, %.1f) voxels"
                      % (d[0] * s, d[1] * s, d[2] * s))

        if kwargs.get('disp', True):
            print("=== MULTISCALE: scale 2^%d (image decimated by %d, h_i=%d/%d=%d) ==="
                  % (iscale, s, h, s, h_i))

        U, res, dU = correlate_gpu(f_i, g_i, Nx, Ny, Nz, h_i, xp, g_origin=g_origin_i,
                                    l0=l0_i, U0=U, **kwargs)
        prev_s = s

        if xp is not np:
            del f_i, g_i
            xp.get_default_memory_pool().free_all_blocks()

    return U, res, dU
