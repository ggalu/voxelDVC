# -*- coding: utf-8 -*-
"""Matrix-free matvec kernels and PCG solver for the GPU DVC Hessian and
regularizers (Phase 1).

All array-heavy functions take an `xp` module (numpy or cupy) so they can
be exercised on CPU for testing and on GPU for production, following the
backend-agnostic guidance in dvc_implementation_guide.md.

Structured-mesh conventions (must match mesher.StructuredMeshHex8 +
Mesh.Connectivity(order='N') + geometry_dvc):

  - Nx, Ny, Nz: number of elements along each axis. Mesh has
    (Nx+1)*(Ny+1)*(Nz+1) nodes, Ndof = 3*nnodes.
  - Node index for node (i,j,k): idx = i + j*(Nx+1) + k*(Nx+1)*(Ny+1)
    (x fastest), DOFs = 3*idx + {0,1,2} (node-major, order='N').
  - Element index e -> (i,j,k) element coords: i = e % Nx,
    j = (e // Nx) % Ny, k = e // (Nx*Ny) (x fastest).
  - Voxel-center quadrature with integer edge length h: the reference
    image / grad_f / mask0 arrays have shape (Nx*h+1, Ny*h+1, Nz*h+1[,...]),
    with voxel index vidx = vx + vy*Nvx + vz*Nvx*Nvy, Nvx = Nx*h+1,
    Nvy = Ny*h+1 (x fastest, same convention as node indexing).
"""
import numpy as np

from .geometry_dvc import build_N_stencil, build_node_index, element_node_ids


# ---------------------------------------------------------------------------
# Implicit connectivity
# ---------------------------------------------------------------------------

def get_elem_dofs(elem_ids, Nx, Ny, xp):
    """DOF indices for a chunk of elements via implicit connectivity.

    Parameters
    ----------
    elem_ids : (chunk,) int array
    Nx, Ny : int -- number of elements along x, y
    xp : numpy or cupy module

    Returns
    -------
    dofs : (chunk, 24) int array, node-major local ordering 0..7 matching
        geometry_dvc's documented Hex8 corner convention.
    """
    sj = Nx + 1            # node stride along y
    si = 1                 # node stride along x
    sk = (Nx + 1) * (Ny + 1)  # node stride along z

    i_e = elem_ids % Nx
    j_e = (elem_ids // Nx) % Ny
    k_e = elem_ids // (Nx * Ny)
    n0 = i_e * si + j_e * sj + k_e * sk  # (chunk,)

    node_offsets = xp.asarray(
        [0, si, si + sj, sj, sk, si + sk, si + sj + sk, sj + sk], dtype=elem_ids.dtype
    )
    nodes = n0[:, None] + node_offsets[None, :]  # (chunk, 8)

    dofs = nodes[:, :, None] * 3 + xp.arange(3, dtype=elem_ids.dtype)[None, None, :]
    return dofs.reshape(-1, 24)  # (chunk, 24)


def get_voxel_offsets(h, Nx, Ny, xp):
    """Flat voxel-index offsets (relative to an element's origin voxel) for
    the (h+1)^3 voxel-center stencil points, in the same order as
    geometry_dvc.build_N_stencil(h). Nvx = Nx*h+1, Nvy = Ny*h+1.

    Returns
    -------
    offsets : ((h+1)^3,) int array
    """
    Nvx = Nx * h + 1
    Nvy = Ny * h + 1
    k = xp.arange(h + 1, dtype=xp.int32)
    qx, qy, qz = xp.meshgrid(k, k, k, indexing='ij')
    offsets = (qx + qy * Nvx + qz * Nvx * Nvy).ravel()
    return offsets


def get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp):
    """Flat voxel index of element (i,j,k)'s origin voxel (q=(0,0,0)),
    Nvx = Nx*h+1, Nvy = Ny*h+1."""
    Nvx = Nx * h + 1
    Nvy = Ny * h + 1
    i_e = elem_ids % Nx
    j_e = (elem_ids // Nx) % Ny
    k_e = elem_ids // (Nx * Ny)
    return h * (i_e + j_e * Nvx + k_e * Nvx * Nvy)


def boundary_element_mask(Nx, Ny, Nz):
    """Per-element mask flagging the outer one-element boundary shell of the
    structured mesh -- every element with an index on a domain face
    (i in {0, Nx-1} or j in {0, Ny-1} or k in {0, Nz-1}).

    These elements' cell strain is systematically biased: the H1/Laplacian
    regularizer imposes a natural (Neumann) zero-normal-gradient boundary
    condition that pulls the recovered normal strain toward zero in the
    boundary layer (see docs/TUTORIAL.md 5.4). The bias is real image data or
    not -- it is a property of the regularized solve, not of overlap loss --
    so the shell is flagged unconditionally as untrustworthy, independent of
    the active/material masks.

    Returns
    -------
    elem_boundary : (Nx*Ny*Nz,) bool array, x-fastest element order
        (elem_id = i + j*Nx + k*Nx*Ny), True on the outer element shell.
    """
    mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    mask[[0, -1], :, :] = True
    mask[:, [0, -1], :] = True
    mask[:, :, [0, -1]] = True
    # x-fastest flatten (elem_id = i + j*Nx + k*Nx*Ny) to match elem_ids.
    return mask.transpose(2, 1, 0).ravel()


def _unsafe_elements(active_vox, h, Nx, Ny, Nz):
    """Shared helper: (elem_voxels, elem_has_unsafe_node, unsafe_node) for
    compute_unsafe_voxel_mask / compute_unsafe_element_mask /
    compute_unsafe_node_mask.

    NB: this helper and the four compute_unsafe_*/compute_overlap_lost_*
    functions below hard-code `numpy` rather than threading an `xp` backend
    parameter through like the rest of this module (matvec_H, pcg, etc.).
    That is intentional, not an oversight: they are post-hoc diagnostics run
    once on CPU after the GPU solve has already produced `active_vox`/`U` as
    plain numpy arrays (see src/write_output.py), so there is no live GPU array
    to operate on here.

    elem_voxels : (Nelem, (h+1)^3) int array, flat voxel indices per element.
    elem_has_unsafe_node : (Nelem,) bool array, True if the element touches
        a node that belongs to some element containing at least one inactive
        voxel.
    unsafe_node : ((Nx+1)*(Ny+1)*(Nz+1),) bool array, True for nodes
        belonging to some element containing at least one inactive voxel
        (node ordering idx = i + j*(Nx+1) + k*(Nx+1)*(Ny+1), matching
        build_node_index).
    """
    elem_ids = np.arange(Nx * Ny * Nz)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, np)
    elem_origins = get_elem_voxel_origins(elem_ids, h, Nx, Ny, np)
    elem_voxels = elem_origins[:, None] + voxel_offsets[None, :]  # (Nelem, (h+1)^3)

    elem_has_inactive = (~active_vox)[elem_voxels].any(axis=1)  # (Nelem,)

    node_idx = build_node_index(Nx, Ny, Nz)
    elem_nodes = element_node_ids(node_idx, Nx, Ny, Nz)  # (Nelem, 8)

    n_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
    unsafe_node = np.zeros(n_nodes, dtype=bool)
    unsafe_node[elem_nodes[elem_has_inactive].ravel()] = True

    elem_has_unsafe_node = unsafe_node[elem_nodes].any(axis=1)  # (Nelem,)

    # Always trim the outer one-element boundary shell (regularizer Neumann-BC
    # strain bias, see docs/TUTORIAL.md 5.4). OR it in *after* the node
    # dilation above so the flagged shell stays exactly one element thick
    # rather than being widened a second layer inward by the dilation.
    elem_boundary = boundary_element_mask(Nx, Ny, Nz)
    elem_has_unsafe_node = elem_has_unsafe_node | elem_boundary
    unsafe_node[elem_nodes[elem_boundary].ravel()] = True
    return elem_voxels, elem_has_unsafe_node, unsafe_node


def compute_unsafe_voxel_mask(active_vox, h, Nx, Ny, Nz):
    """Per-voxel mask flagging voxels affected by "unsafe" nodes.

    A node is unsafe if it belongs to an element that contains at least one
    inactive voxel (`active_vox[vidx] == False`, e.g. a voxel outside the
    overlap region after the dynamic-mask in_bounds check). A voxel is
    "affected" if it belongs to any element that has at least one unsafe
    node -- i.e. the one-element dilation of the inactive-voxel region.

    The outer one-element boundary shell is *always* flagged as well,
    independent of the active mask: its strain/displacement gradient is biased
    by the regularizer's Neumann natural boundary condition (see
    boundary_element_mask and docs/TUTORIAL.md 5.4).

    Parameters
    ----------
    active_vox : (Nvox,) bool array (numpy), True = active voxel.
    h, Nx, Ny, Nz : int -- mesh parameters.

    Returns
    -------
    affected_vox : (Nvox,) bool array, True where the voxel is part of an
        element touching an unsafe node.
    """
    elem_voxels, elem_has_unsafe_node, _ = _unsafe_elements(active_vox, h, Nx, Ny, Nz)
    affected_vox = np.zeros(active_vox.shape[0], dtype=bool)
    affected_vox[elem_voxels[elem_has_unsafe_node].ravel()] = True
    return affected_vox


def compute_unsafe_element_mask(active_vox, h, Nx, Ny, Nz):
    """Per-element mask flagging "unsafe" elements -- elements that touch a
    node belonging to some element containing at least one inactive voxel
    (i.e. the elements `compute_unsafe_voxel_mask`'s voxel dilation is
    derived from).

    Parameters
    ----------
    active_vox : (Nvox,) bool array (numpy), True = active voxel.
    h, Nx, Ny, Nz : int -- mesh parameters.

    Returns
    -------
    elem_has_unsafe_node : (Nx*Ny*Nz,) bool array, x-fastest element order
        (elem_id = i + j*Nx + k*Nx*Ny), True if the element is "unsafe".
    """
    _, elem_has_unsafe_node, _ = _unsafe_elements(active_vox, h, Nx, Ny, Nz)
    return elem_has_unsafe_node


def compute_unsafe_node_mask(active_vox, h, Nx, Ny, Nz):
    """Per-node mask flagging "unsafe" nodes -- nodes belonging to some
    element that contains at least one inactive voxel (outside the final
    dynamic mask).

    Parameters
    ----------
    active_vox : (Nvox,) bool array (numpy), True = active voxel.
    h, Nx, Ny, Nz : int -- mesh parameters.

    Returns
    -------
    unsafe_node : ((Nx+1)*(Ny+1)*(Nz+1),) bool array, node ordering
        idx = i + j*(Nx+1) + k*(Nx+1)*(Ny+1) (matches build_node_index /
        the nodal U_recovered_grid layout), True if the node is "unsafe".
    """
    _, _, unsafe_node = _unsafe_elements(active_vox, h, Nx, Ny, Nz)
    return unsafe_node


def compute_overlap_lost_element_mask(active_vox, mask0_vox, h, Nx, Ny, Nz,
                                      frac_threshold=0.0, min_material_frac=0.5):
    """Per-element mask flagging elements whose cell strain is unreliable,
    combining three independent criteria: *material lost correlation overlap*,
    *too little material to measure strain* (both excluding mere
    well-correlated porosity), and the *outer one-element boundary shell*
    (whose normal strain is biased by the regularizer regardless of material,
    see criterion 3 / boundary_element_mask).

    `compute_unsafe_element_mask` treats every inactive voxel as a defect,
    including no-material / pore voxels (`mask0_vox == False`), so in a
    porous specimen it flags almost the whole volume (the pore network
    dilated by an element). That is the wrong question for strain validity:
    a pore inside an element does not invalidate the element's strain if the
    surrounding *material* correlated well -- the displacement is interpolated
    from nodes that the material constrains.

    An element is flagged unsafe if EITHER:

    1. **Overlap loss.** It contains material and the lost fraction of its
       material voxels exceeds `frac_threshold`, where a "lost" voxel is
       material in the reference (`mask0_vox == True`) yet dropped from the
       correlation (`active_vox == False`, i.e. the deformed position x+u(x)
       left the deformed-image bounds or landed on g<=0). This criterion is
       1-element dilated through shared nodes (an element is also flagged if
       it shares a node with an overlap-lost element), because the shared
       nodes' displacement -- and hence this element's centroid strain -- is
       then under-constrained. This matches the node->element dilation
       `compute_unsafe_element_mask` uses, so the two masks are comparable.

    2. **Insufficient material (count-fraction floor).** Its material fraction
       -- the fraction of its (h+1)^3 quadrature voxels that are material,
       `(# mask0_vox==True) / (h+1)^3` -- is below `min_material_frac`. Such
       an element is mostly pore/air, so its strain is governed by
       interpolation/regularization from neighbouring nodes rather than by
       image data inside the element, i.e. it is not a real strain
       *measurement* even if no material lost overlap. This criterion is
       per-element (NOT dilated): it is a local "is there enough material here
       to measure strain" test. Note `mask0_vox = (ref > 0)` is the existing
       binary material/no-material classification (zero gray level = no
       material/no contrast); the floor is a threshold on the per-element
       COUNT fraction of such voxels, not a gray-level threshold. On a fully
       dense specimen (`mask0_vox` all True, e.g. the elastic dataset) this
       criterion never triggers, so the mask reduces to criteria 1 and 3.

    Returns the union of 1, 2 and 3. Its complement -- interior elements with
    enough material AND no lost material overlap -- is the set where the cell
    strain is a trustworthy measurement.

    Parameters
    ----------
    active_vox : (Nvox,) bool array (numpy), True = active voxel
        (mask0 & in_bounds(x+u) & g(x+u)>0, e.g. from compute_active_mask).
    mask0_vox : (Nvox,) bool array (numpy), True = reference voxel is
        material (f_vox > 0), same convention as correlate_gpu's mask0_vox.
    h, Nx, Ny, Nz : int -- mesh parameters.
    frac_threshold : float in [0, 1), default 0.0
        An element is overlap-lost (criterion 1) if (lost material voxels)/
        (material voxels) > frac_threshold. 0.0 flags an element as soon as
        any of its material voxels lost overlap; raise it to tolerate a thin
        lost shell.
    min_material_frac : float in [0, 1], default 0.5
        Count-fraction floor (criterion 2): an element is flagged if fewer
        than this fraction of its (h+1)^3 voxels are material. Default 0.5
        keeps only elements that are at least half material. Set to 0.0 to
        disable the floor and recover the pure overlap-loss mask.

    Returns
    -------
    elem_unsafe : (Nx*Ny*Nz,) bool array, x-fastest element order
        (elem_id = i + j*Nx + k*Nx*Ny), True if the element's strain may be
        unreliable (lost material overlap OR too little material OR on the
        outer boundary shell). Its complement is the set of elements where the
        cell strain is trustworthy.
    """
    elem_ids = np.arange(Nx * Ny * Nz)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, np)
    elem_origins = get_elem_voxel_origins(elem_ids, h, Nx, Ny, np)
    elem_voxels = elem_origins[:, None] + voxel_offsets[None, :]  # (Nelem, (h+1)^3)
    n_per_elem = elem_voxels.shape[1]                             # (h+1)^3

    # Criterion 1: material lost overlap (dilated through shared nodes).
    lost_vox = mask0_vox & ~active_vox            # material that lost overlap
    n_material = mask0_vox[elem_voxels].sum(axis=1)
    n_lost = lost_vox[elem_voxels].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        lost_frac = np.where(n_material > 0, n_lost / n_material, 0.0)
    elem_lost = (n_material > 0) & (lost_frac > frac_threshold)

    node_idx = build_node_index(Nx, Ny, Nz)
    elem_nodes = element_node_ids(node_idx, Nx, Ny, Nz)  # (Nelem, 8)
    n_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
    node_lost = np.zeros(n_nodes, dtype=bool)
    node_lost[elem_nodes[elem_lost].ravel()] = True
    elem_overlap_unsafe = node_lost[elem_nodes].any(axis=1)

    # Criterion 2: count-fraction floor on material (per-element, not dilated).
    material_frac = n_material / n_per_elem
    elem_sparse = material_frac < min_material_frac

    # Criterion 3: outer one-element boundary shell -- its cell strain is
    # biased toward zero normal strain by the regularizer's Neumann natural BC
    # regardless of material/overlap (see boundary_element_mask / TUTORIAL 5.4).
    elem_boundary = boundary_element_mask(Nx, Ny, Nz)

    return elem_overlap_unsafe | elem_sparse | elem_boundary


# ---------------------------------------------------------------------------
# Constant/per-element-scalar-modulus matvec via a precomputed K_ref
# (used for Laplacian and elastic/equilibrium-gap regularizers)
# ---------------------------------------------------------------------------

def matvec_K_ref(v, K_ref, E_elem, Nx, Ny, Nz, xp, chunk_size=500_000):
    """f = K @ v, K = sum_e E_elem[e] * (P_e^T K_ref P_e), matrix-free.

    Parameters
    ----------
    v : (Ndof,) array
    K_ref : (24, 24) array
    E_elem : (Ne,) array or scalar -- per-element modulus/scale
    Nx, Ny, Nz : int -- elements per axis
    """
    Ndof = v.shape[0]
    Ne = Nx * Ny * Nz
    f = xp.zeros(Ndof, dtype=v.dtype)
    K_ref_x = xp.asarray(K_ref, dtype=v.dtype)
    E_elem_arr = xp.broadcast_to(xp.asarray(E_elem, dtype=v.dtype), (Ne,))

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        dofs = get_elem_dofs(elem_ids, Nx, Ny, xp)  # (chunk, 24)
        v_loc = v[dofs]                              # (chunk, 24)
        Kv_loc = xp.einsum('ij,kj->ki', K_ref_x, v_loc)
        Kv_loc *= E_elem_arr[start:end, None]
        xp.add.at(f, dofs.ravel(), Kv_loc.ravel())

    return f


def build_diagonal_K_ref(K_ref, E_elem, Nx, Ny, Nz, Ndof, xp):
    """Diagonal of the matrix-free K_ref operator (for Jacobi precond)."""
    diag_ref = xp.diag(xp.asarray(K_ref))  # (24,)
    Ne = Nx * Ny * Nz
    E_elem_arr = xp.broadcast_to(xp.asarray(E_elem), (Ne,))
    diag = xp.zeros(Ndof, dtype=diag_ref.dtype)

    chunk_size = 500_000
    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        dofs = get_elem_dofs(elem_ids, Nx, Ny, xp)  # (chunk, 24)
        contrib = E_elem_arr[start:end, None] * diag_ref[None, :]
        xp.add.at(diag, dofs.ravel(), contrib.ravel())

    return diag


# ---------------------------------------------------------------------------
# Equilibrium-gap regularizer: R_m = K_i^T K_i = K @ P_i @ K (Section 7,
# Phase 3). K is symmetric and P_i is a diagonal 0/1 interior-DOF projector,
# so R_m @ v = K @ (P_i * (K @ v)) -- two matvec_K_ref calls with an
# elementwise mask multiply in between, no (P_i K) ever assembled.
# ---------------------------------------------------------------------------

def matvec_equilibrium_gap(v, K_ref_elastic, interior_mask, E_elem, Nx, Ny, Nz,
                             xp, chunk_size=500_000):
    """f = R_m @ v = K @ (P_i * (K @ v)), matrix-free.

    Parameters
    ----------
    v : (Ndof,) array
    K_ref_elastic : (24, 24) array -- geometry_dvc.build_K_ref(nu)
    interior_mask : (Ndof,) bool/float array -- P_i diagonal,
        geometry_dvc.build_interior_dof_mask(Nx, Ny, Nz)
    E_elem : (Ne,) array or scalar -- per-element modulus/scale (use
        E_elem=h for a mesh of integer edge length h, see Section 7)
    """
    w = matvec_K_ref(v, K_ref_elastic, E_elem, Nx, Ny, Nz, xp, chunk_size)
    w = interior_mask * w
    return matvec_K_ref(w, K_ref_elastic, E_elem, Nx, Ny, Nz, xp, chunk_size)


def build_diagonal_equilibrium_gap(K_ref_elastic, interior_mask, E_elem, Nx, Ny, Nz,
                                     Ndof, xp, n_probes=32, seed=0):
    """Stochastic (Hutchinson) estimate of diag(R_m), for a Jacobi
    preconditioner. R_m = K @ P_i @ K is never assembled, so its diagonal
    cannot be read off element-local contributions like build_diagonal_H/
    build_diagonal_K_ref; instead, diag(R_m)_i ~= E[z_i (R_m z)_i] over
    Rademacher (+-1) probe vectors z, which is unbiased
    (E[z_i z_j] = delta_ij). n_probes ~ 32 is sufficient for a
    preconditioner (does not need to be exact).
    """
    rng = np.random.default_rng(seed)
    diag_acc = xp.zeros(Ndof, dtype=xp.float64)
    for _ in range(n_probes):
        z = xp.asarray(rng.choice([-1.0, 1.0], size=Ndof))
        Rz = matvec_equilibrium_gap(z, K_ref_elastic, interior_mask, E_elem, Nx, Ny, Nz, xp)
        diag_acc = diag_acc + z * Rz
    diag = diag_acc / n_probes
    return xp.clip(diag, 1e-12 * float(xp.abs(diag).max()), None)


# ---------------------------------------------------------------------------
# Matrix-free DVC Hessian: H @ v = sum_e sum_q w_q*mask0_q * v_eq (v_eq . v_loc)
# ---------------------------------------------------------------------------

def auto_chunk_size(nq, max_tensor_elems=20_000_000):
    """Element-chunk size such that the (chunk, nq, 24) v_eq tensor has at
    most `max_tensor_elems` entries. nq = (h+1)^3 grows quickly with the
    voxel-center stencil edge length h, so the chunk size must shrink
    accordingly to bound memory while still vectorizing the q loop."""
    return max(1, max_tensor_elems // (nq * 24))


def matvec_H(v, grad_f_vox, mask0_vox, N_stencil, w_stencil, h, Nx, Ny, Nz,
              xp, chunk_size=None):
    """f = H @ v, the matrix-free DVC Hessian (Section 2 of
    dvc_gpu_matrixfree_plan.md).

    Parameters
    ----------
    v : (Ndof,) array
    grad_f_vox : (Nvx*Nvy*Nvz, 3) array -- reference-image gradient at each
        voxel, flattened with vidx = vx + vy*Nvx + vz*Nvx*Nvy,
        Nvx=Nx*h+1, Nvy=Ny*h+1, Nvz=Nz*h+1.
    mask0_vox : (Nvx*Nvy*Nvz,) array -- reference-side validity mask.
    N_stencil : ((h+1)^3, 8) array
    w_stencil : ((h+1)^3,) array
    h, Nx, Ny, Nz : int
    chunk_size : int, optional
        Number of elements per chunk. Defaults to auto_chunk_size(nq).
    """
    Ndof = v.shape[0]
    Ne = Nx * Ny * Nz
    f = xp.zeros(Ndof, dtype=v.dtype)

    N_stencil_x = xp.asarray(N_stencil, dtype=v.dtype)   # ((h+1)^3, 8)
    w_stencil_x = xp.asarray(w_stencil, dtype=v.dtype)   # ((h+1)^3,)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, xp)     # ((h+1)^3,)
    nq = voxel_offsets.shape[0]
    if chunk_size is None:
        chunk_size = auto_chunk_size(nq)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        dofs = get_elem_dofs(elem_ids, Nx, Ny, xp)               # (chunk, 24)
        v_loc = v[dofs]                                          # (chunk, 24)

        vox0 = get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp)    # (chunk,)
        vox_ids = vox0[:, None] + voxel_offsets[None, :]          # (chunk, nq)

        gradf_q = grad_f_vox[vox_ids]                             # (chunk, nq, 3)
        mask_q = mask0_vox[vox_ids]                               # (chunk, nq)

        # Separable (Kronecker) contraction: v_eq[c,q,3a+k] = N[q,a]*gradf[c,q,k]
        # is never materialized (it would be the dominant (chunk, nq, 24)
        # tensor).  H@v = sum_q w_q mask_q (v_eq . v_loc) v_eq factors through
        # intermediates no larger than (chunk, nq, 3).
        v_loc3 = v_loc.reshape(-1, 8, 3)                          # (chunk, 8, 3)
        inner = xp.einsum('qa,cak->cqk', N_stencil_x, v_loc3)     # (chunk, nq, 3)
        dot = xp.einsum('cqk,cqk->cq', gradf_q, inner)            # (chunk, nq)
        coeff = w_stencil_x[None, :] * mask_q * dot               # (chunk, nq)
        tmp = coeff[:, :, None] * gradf_q                         # (chunk, nq, 3)
        Hv_loc = xp.einsum('qa,cqk->cak', N_stencil_x, tmp).reshape(-1, 24)

        xp.add.at(f, dofs.ravel(), Hv_loc.ravel())

    return f


def build_diagonal_H(grad_f_vox, mask0_vox, N_stencil, w_stencil, h, Nx, Ny, Nz,
                       Ndof, xp, chunk_size=None):
    """Diagonal of the matrix-free H operator (for Jacobi precond):
    diag[i] = sum over (e,q) touching dof i of w_q*mask0_q*v_eq[i]^2."""
    diag = xp.zeros(Ndof, dtype=grad_f_vox.dtype)
    Ne = Nx * Ny * Nz

    N_stencil_x = xp.asarray(N_stencil, dtype=grad_f_vox.dtype)
    N_stencil_sq = N_stencil_x ** 2
    w_stencil_x = xp.asarray(w_stencil, dtype=grad_f_vox.dtype)
    voxel_offsets = get_voxel_offsets(h, Nx, Ny, xp)
    nq = voxel_offsets.shape[0]
    if chunk_size is None:
        chunk_size = auto_chunk_size(nq)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = xp.arange(start, end, dtype=xp.int32)
        dofs = get_elem_dofs(elem_ids, Nx, Ny, xp)

        vox0 = get_elem_voxel_origins(elem_ids, h, Nx, Ny, xp)
        vox_ids = vox0[:, None] + voxel_offsets[None, :]

        gradf_q = grad_f_vox[vox_ids]
        mask_q = mask0_vox[vox_ids]

        # diag(v_eq v_eq^T)[c,3a+k] = N[q,a]^2 * gradf[c,q,k]^2, summed over q
        # with weight w_q*mask_q -- separable, so the (chunk, nq, 24) v_eq
        # tensor is never built (largest intermediate is (chunk, nq, 3)).
        tmp = (w_stencil_x[None, :] * mask_q)[:, :, None] * gradf_q ** 2  # (chunk, nq, 3)
        diag_loc = xp.einsum('qa,cqk->cak', N_stencil_sq, tmp).reshape(-1, 24)

        xp.add.at(diag, dofs.ravel(), diag_loc.ravel())

    return diag


# ---------------------------------------------------------------------------
# Preconditioned Conjugate Gradient
# ---------------------------------------------------------------------------

def pcg(matvec, b, M_inv, xp, x0=None, max_iter=1000, tol=1e-8, verbose=False):
    """Matrix-free PCG. matvec: v -> A@v. M_inv: array (Ndof,) diagonal
    preconditioner inverse."""
    x = xp.zeros_like(b) if x0 is None else x0.copy()

    r = b - matvec(x)
    z = M_inv * r
    p = z.copy()
    rz = xp.dot(r, z)
    r0 = xp.dot(b, b) + 1e-30

    for it in range(max_iter):
        Ap = matvec(p)
        pAp = xp.dot(p, Ap)
        alpha = rz / (pAp + 1e-30)

        x = x + alpha * p
        r = r - alpha * Ap

        res = float(xp.sqrt(xp.dot(r, r) / r0))
        if verbose and (it % 10 == 0 or it < 5):
            print(f"  PCG iter {it:4d}  rel.residual = {res:.3e}")
        if res < tol:
            if verbose:
                print(f"  Converged at iter {it}, residual = {res:.3e}")
            break

        z = M_inv * r
        rz_new = xp.dot(r, z)
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new

    return x
