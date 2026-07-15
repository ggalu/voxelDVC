# -*- coding: utf-8 -*-
"""Matrix-free geometric primitives for the GPU DVC solver (Phase 0).

Ported/adapted from /home/gcg/Coding/voxelFEM/geometry.py. Provides:

  - build_K_ref(nu): 24x24 reference Hex8 stiffness for a unit cube, E=1.
  - build_N_stencil(h): Hex8 shape functions evaluated at the (h+1)^3
    voxel-center quadrature points (VoxelCenterStencil), shape
    ((h+1)^3, 8).
  - build_node_index / element_node_ids: implicit node/DOF indexing for a
    structured Hex8 mesh, matching the node ordering produced by
    mesher.StructuredMeshHex8 and mesh.Mesh.Connectivity(order='N').

Local node ordering (shared by mesh.py's Hex8 shape functions, mesher.py's
element connectivity, and voxelFEM's build_K_ref):
    0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
    4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
"""
import numpy as np


# ---------------------------------------------------------------------------
# Voxel-center quadrature rule
# ---------------------------------------------------------------------------

def VoxelCenterStencil(h):
    """Natural coordinates and trapezoidal weights for a voxel-center
    integration rule over a single Hex8 element of edge length h voxels.

    Places (h+1)^3 points at every integer voxel center from 0 to h along
    each local axis (xi = 2*k/h - 1, k = 0..h), so points coincide with the
    image grid and no interpolation is needed to sample the reference image.

    Points on element faces/edges/corners (k in {0, h}) are shared with
    neighboring elements and get trapezoidal weight 1/2 per axis (1/2, 1/4,
    1/8 for face/edge/corner points), so that each voxel of the global image
    grid receives total weight 1 when contributions from all elements
    sharing it are summed.
    """
    k = np.arange(h + 1)
    xi1d = 2 * k / h - 1
    w1d = np.ones(h + 1)
    w1d[0] = 0.5
    w1d[-1] = 0.5
    X, Y, Z = np.meshgrid(xi1d, xi1d, xi1d, indexing='ij')
    WX, WY, WZ = np.meshgrid(w1d, w1d, w1d, indexing='ij')
    xg = X.ravel()
    yg = Y.ravel()
    zg = Z.ravel()
    wg = (WX * WY * WZ).ravel()
    return xg, yg, zg, wg


# ---------------------------------------------------------------------------
# Reference element stiffness for a unit cube (Q1 trilinear hex)
# ---------------------------------------------------------------------------

def build_K_ref(nu: float) -> np.ndarray:
    """24x24 reference element stiffness matrix for a unit cube, E=1,
    using full 2x2x2 Gauss quadrature. DOF ordering is node-major:
    node 0 (u0,v0,w0), node 1 (u1,v1,w1), ..., node 7 (u7,v7,w7).
    """
    gp = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    gw = np.array([1.0, 1.0])

    c1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c3 = 1.0 / (2.0 * (1.0 + nu))
    D = np.array([
        [c1, c2, c2,  0,  0,  0],
        [c2, c1, c2,  0,  0,  0],
        [c2, c2, c1,  0,  0,  0],
        [ 0,  0,  0, c3,  0,  0],
        [ 0,  0,  0,  0, c3,  0],
        [ 0,  0,  0,  0,  0, c3],
    ])

    K = np.zeros((24, 24))
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                w = 1.0
                B, detJ = _B_matrix(xi, eta, zeta)
                K += w * detJ * (B.T @ D @ B)
    return K


def build_K_ref_laplacian() -> np.ndarray:
    """24x24 reference "vector Laplacian" matrix for a unit cube, matching
    mesh.Mesh.Laplacian(): for each displacement component c (u,v,w)
    independently, L_c = sum_d (dN/dx_d)^T W (dN/dx_d) over d in {x,y,z}
    (full 2x2x2 Gauss quadrature). The 24x24 result is block-structured in
    node-major DOF order: entry [3a+c, 3b+c] = L8[a,b], zero for c != c'.
    """
    gp = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    detJ = 0.125

    L8 = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                w = 1.0
                xi_, eta_, zeta_ = 0.5*(xi+1.0), 0.5*(eta+1.0), 0.5*(zeta+1.0)
                dN = _shape_grads(xi_, eta_, zeta_)  # (3, 8): dN/dx,dN/dy,dN/dz
                L8 += w * detJ * (dN.T @ dN)

    K = np.zeros((24, 24))
    for c in range(3):
        K[c::3, c::3] = L8
    return K


def _shape_grads(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Gradients of Q1 shape functions w.r.t. physical (x,y,z) in [0,1]^3,
    given natural coords (xi,eta,zeta) in [0,1]^3. Returns (3, 8)."""
    xi1, eta1, zeta1 = 1 - xi, 1 - eta, 1 - zeta
    dN = np.array([
        [-eta1*zeta1,  eta1*zeta1,  eta*zeta1, -eta*zeta1,
         -eta1*zeta,   eta1*zeta,   eta*zeta,  -eta*zeta],
        [-xi1*zeta1,  -xi*zeta1,    xi*zeta1,   xi1*zeta1,
         -xi1*zeta,   -xi*zeta,     xi*zeta,    xi1*zeta],
        [-xi1*eta1,   -xi*eta1,    -xi*eta,    -xi1*eta,
          xi1*eta1,    xi*eta1,     xi*eta,     xi1*eta],
    ])
    return dN


def _B_matrix(xi: float, eta: float, zeta: float):
    """Strain-displacement matrix B (6x24) and detJ for a unit cube,
    given Gauss coordinates (xi,eta,zeta) in [-1,1]."""
    xi_, eta_, zeta_ = 0.5 * (xi + 1.0), 0.5 * (eta + 1.0), 0.5 * (zeta + 1.0)
    detJ = 0.125

    dN = _shape_grads(xi_, eta_, zeta_)

    B = np.zeros((6, 24))
    for a in range(8):
        col = 3 * a
        B[0, col] = dN[0, a]
        B[1, col+1] = dN[1, a]
        B[2, col+2] = dN[2, a]
        B[3, col] = dN[1, a]
        B[3, col+1] = dN[0, a]
        B[4, col+1] = dN[2, a]
        B[4, col+2] = dN[1, a]
        B[5, col] = dN[2, a]
        B[5, col+2] = dN[0, a]
    return B, detJ


# ---------------------------------------------------------------------------
# Voxel-center stencil shape functions (N_stencil)
# ---------------------------------------------------------------------------

def _hex8_N(x, y, z):
    """Hex8 shape functions at natural coordinates (x,y,z) in [-1,1]^3,
    matching mesh.ShapeFunctions's N for eltype=5. x,y,z: 1D arrays of
    length npts. Returns (npts, 8)."""
    return 0.125 * np.stack([
        (1-x)*(1-y)*(1-z),
        (1+x)*(1-y)*(1-z),
        (1+x)*(1+y)*(1-z),
        (1-x)*(1+y)*(1-z),
        (1-x)*(1-y)*(1+z),
        (1+x)*(1-y)*(1+z),
        (1+x)*(1+y)*(1+z),
        (1-x)*(1+y)*(1+z),
    ], axis=-1)


def build_N_stencil(h: int):
    """Hex8 shape functions evaluated at the (h+1)^3 voxel-center
    quadrature points of VoxelCenterStencil(h).

    Returns
    -------
    N_stencil : (h+1)^3 x 8 array
        N_stencil[q, a] = N_a(xg_q, yg_q, zg_q).
    w_stencil : (h+1)^3 array
        Trapezoidal quadrature weights from VoxelCenterStencil.
    """
    xg, yg, zg, wg = VoxelCenterStencil(h)
    N_stencil = _hex8_N(xg, yg, zg)
    return N_stencil, wg


# ---------------------------------------------------------------------------
# Implicit node/DOF indexing for a structured Hex8 mesh
# ---------------------------------------------------------------------------

def build_node_index(Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """Global node index for node (i,j,k), matching the node ordering
    produced by mesher.StructuredMeshHex8 (x fastest, then y, then z):
        idx = i + j*(Nx+1) + k*(Nx+1)*(Ny+1)
    Returns shape (Nx+1, Ny+1, Nz+1) int array.
    """
    Nn = (Nx + 1) * (Ny + 1) * (Nz + 1)
    return np.arange(Nn).reshape(Nz + 1, Ny + 1, Nx + 1).transpose(2, 1, 0)


def build_interior_dof_mask(Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """Boolean mask of length Ndof=3*(Nx+1)*(Ny+1)*(Nz+1) selecting DOFs of
    interior nodes, i.e. nodes that do not lie on the outer bounding-box
    surface of the structured mesh (matches mesh.Mesh.InteriorDofMask() for
    a mesh spanning [0,Nx]x[0,Ny]x[0,Nz] element coordinates, used to build
    P_i for the equilibrium-gap regularizer).
    """
    node_idx = build_node_index(Nx, Ny, Nz)
    interior = np.zeros_like(node_idx, dtype=bool)
    interior[1:Nx, 1:Ny, 1:Nz] = True

    mask = np.zeros(3 * node_idx.size, dtype=bool)
    flat_idx = node_idx.ravel()
    interior_flat = interior.ravel()
    for c in range(3):
        mask[3 * flat_idx + c] = interior_flat
    return mask


def element_node_ids(node_idx: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """For each element (i,j,k), return its 8 node global IDs, in the
    local node ordering documented at the top of this module (matches
    mesher.StructuredMeshHex8's element connectivity and build_K_ref).
    Returns shape (Nx*Ny*Nz, 8), row r = i + j*Nx + k*Nx*Ny (x fastest,
    matching the elem_id convention used by get_elem_voxel_origins /
    get_voxel_offsets).
    """
    conn = np.stack([
        node_idx[:Nx,    :Ny,    :Nz],
        node_idx[1:Nx+1, :Ny,    :Nz],
        node_idx[1:Nx+1, 1:Ny+1, :Nz],
        node_idx[:Nx,    1:Ny+1, :Nz],
        node_idx[:Nx,    :Ny,    1:Nz+1],
        node_idx[1:Nx+1, :Ny,    1:Nz+1],
        node_idx[1:Nx+1, 1:Ny+1, 1:Nz+1],
        node_idx[:Nx,    1:Ny+1, 1:Nz+1],
    ], axis=-1)  # shape (Nx, Ny, Nz, 8)
    # Flatten (Nx,Ny,Nz) x-fastest -> r = i + j*Nx + k*Nx*Ny.
    return conn.transpose(2, 1, 0, 3).reshape(-1, 8)
