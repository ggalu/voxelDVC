# voxelDVC Tutorial

A walkthrough of the matrix-free, GPU-accelerated Digital Volume Correlation
(DVC) pipeline in this repository, built around the `voxeldvc` console-script
subcommands (see `src/voxeldvc/`) and the tests in `tests/`. It assumes
you've read the formulation summary in the top-level `README.md`; this
document focuses on *how to actually drive the code* — what to call, with
what parameters, and what comes out.

## 1. What the code does

Given two 3D volumes — a reference image `f` and a deformed image `g` of the
same specimen before/after loading — the pipeline recovers a displacement
field `u(x)` such that `f(x) = g(x + u(x))`, on a structured Hex8 finite
element mesh. The field is regularized (Laplacian or equilibrium-gap) and
solved with Gauss-Newton + Jacobi-preconditioned CG, entirely matrix-free on
the GPU (CuPy). There is no CPU/SciPy assembly path — `cupy` is required.

## 2. Installation

```bash
mamba create -n voxeldvc python=3.12 -y
mamba activate voxeldvc
cd /path/to/voxelDVC
pip install -e ".[gpu,viz]"
```

(`conda` works the same as `mamba` here.) This installs the `voxeldvc`
console script plus:

- **`gpu`** → `cupy-cuda12x[ctk]`. The `[ctk]` matters: bare `cupy-cuda12x`
  bundles the CUDA *runtime* libs but not the headers/`nvrtc` this project's
  raw kernels need to JIT-compile, and fails at the first kernel call with
  `RuntimeError: Failed to find CUDA headers`. Pick a different CuPy wheel
  (e.g. `cupy-cuda11x`) in `pyproject.toml` if your driver doesn't support
  CUDA 12 — GPU support itself is required, there is no CPU fallback.
- **`viz`** → `napari[pyqt]`. Bare `napari` has no Qt backend installed, so
  `voxeldvc view-*` can't open a window without it.

Verify the install:

```bash
voxeldvc --help
```

## 3. Quick start

The complete workflow for a real dataset is two commands run from the
repository root, demonstrated here on the PA6GF30 CT pair bundled in
`assets/`.

### Step 1 — Preprocess

```bash
voxeldvc preprocess assets/PA6GF30_0.npy assets/PA6GF30_1.npy --output-dir test
```

`voxeldvc preprocess` estimates the affine alignment between the two volumes,
finds the valid overlap region, snaps it to an integer multiple of `h`
voxels per axis, and writes three `.npy` files plus a JSON into `test/`.

The affine is found in two stages: a coarse, robust `affine_prealign` (a
9-DOF ZNCC/Powell search on a `decimate=4` subsample — good for large rigid
offsets), followed by a **full-resolution Gauss-Newton refinement**
(`affine_refine_gn`). The refinement matters because a few-percent normal
strain is sub-voxel once the volume is decimated, so the coarse stage can
miss it entirely — on the elastic ground-truth pair it recovers `s_zz≈1.003`
where the true value is `≈0.95`. The refinement is analytic Gauss-Newton with
the exact 12-parameter affine Jacobian, run through a Gaussian scale-space
pyramid; it warps *ref* onto *def* (the generative direction, so the fit can
drive the residual to near zero) and converts the result back into the
`(A, t, c)` the downstream expects. On the ground-truth pair it drops the
alignment residual ~94% and recovers the affine essentially exactly. Disable
it with `--no-affine-refine` only for debugging.

Condensed output:

```
Running affine_prealign (decimate=4) …
  A =
[[ 1.02689  0.00079  0.00048]
 [-0.00082  0.98643  0.00273]
 [-0.00049 -0.00272  0.99013]]
  rotation (deg) xyz   = (-0.158, 0.027, -0.046)
  scale    (x,y,z)     = (1.0269, 0.9901, 0.9864)
  translation t (vox)  = (-18.949, 1.011, 12.896)
  center c (vox)       = (96.000, 96.000, 96.000)

Final crop shape : (169, 193, 177)
  axis 0 : 169 = 21*8+1  ✓
  axis 1 : 193 = 24*8+1  ✓
  axis 2 : 177 = 22*8+1  ✓

Deformed bounding box for correlation:
  lo_def = [0, 0, 14]  hi_def = [176, 193, 193]
  def_preprocessed shape = (176, 193, 179)

Saved test/ref_preprocessed.npy         shape=(169, 193, 177)  dtype=uint16
Saved test/def_preprocessed.npy         shape=(176, 193, 179)  dtype=uint16
Saved test/def_ref_aligned.npy          shape=(169, 193, 177)  dtype=uint16
Saved test/affine_prealign.json
```

Three images are written:

| File | Contents |
|------|----------|
| `ref_preprocessed.npy` | Reference cropped to the valid overlap region; shape `(169,193,177)` here |
| `def_preprocessed.npy` | Deformed cropped to the affine-mapped bbox `[def_lo, def_hi)` — larger than ref crop, contains every deformed voxel the DVC will sample |
| `def_ref_aligned.npy` | Deformed at the same index range as the reference — same shape as `ref_preprocessed.npy`, used for visualization only |
| `affine_prealign.json` | Affine matrix `A`, translation `t`, center `c`, crop indices, and all metadata needed by `run_dvc.py` |

The affine here captures an ~18-voxel rigid shift in x plus 1–3 % per-axis
scale corrections; 83 % of the warped deformed image is above the gray-level
threshold, confirming good overlap.

**Gray-level threshold (`--glt`).** Voxels whose intensity is at or below
`--glt` (default `0.0`) are treated as *no material / no contrast* and carry
no correlation signal: they are excluded from the overlap crop here and from
the mask, the H/b assembly, and the ZNCC normalization in `voxeldvc run`.
The chosen value is written to `affine_prealign.json` as `glt` and read back
by `run`, so you set it once at preprocess time. The default `0.0` reproduces
the historical "nonzero-intensity is material" convention; raise it when the
background/air sits at a small nonzero gray level (e.g. reconstruction noise
or a padded-with-low-value volume) so that background isn't mistaken for
material.

### Step 2 — Run DVC

```bash
voxeldvc run test
```

The solver runs three coarse-to-fine scales (`2^2 → 2^1 → 2^0`), carrying
the displacement field forward between levels.  The affine prealignment is
baked in as `ext_affine` so the coarsest level starts with a good initial
guess rather than zero.

Condensed output:

```
Mesh: Nx_e=21, Ny_e=24, Nz_e=22  (h=8, ref shape=(169, 193, 177), deformed shape=(176, 193, 179))

GPU matrix-free DVC (multiscale, ext_affine U0, original deformed) …
External affine: t = (3.1, 1.0, 0.9) voxels, c = (74.0, 96.0, 94.0)
=== MULTISCALE: scale 2^2 (image decimated by 4, h_i=8/4=2) ===
...
=== MULTISCALE: scale 2^1 (image decimated by 2, h_i=8/2=4) ===
...
=== MULTISCALE: scale 2^0 (image decimated by 1, h_i=8/1=8) ===
Regularization: laplacian with lambda = 4.735e+08
Iter #  1 | std(res)=3444.13 gl | dU/U=1.14e-01 | active pts=5737149/5773209 (99.4%)
Iter #  2 | std(res)=2738.67 gl | dU/U=3.29e-02 | active pts=5742233/5773209 (99.5%)
Iter #  3 | std(res)=2675.76 gl | dU/U=1.21e-02 | active pts=5742589/5773209 (99.5%)
Iter #  4 | std(res)=2665.02 gl | dU/U=5.98e-03 | active pts=5742711/5773209 (99.5%)
Iter #  5 | std(res)=2663.88 gl | dU/U=3.15e-03 | active pts=5742686/5773209 (99.5%)
Iter #  6 | std(res)=2662.86 gl | dU/U=1.71e-03 | active pts=5742716/5773209 (99.5%)
Iter #  7 | std(res)=2662.92 gl | dU/U=9.39e-04 | active pts=5742699/5773209 (99.5%)

  ux: mean=3.1314  std=1.2464  min=0.8276  max=5.5353
  uy: mean=1.6458  std=0.4714  min=0.7268  max=2.5058
  uz: mean=14.9304  std=0.4273  min=14.1752  max=15.7001
Nonzero (active) residual voxels: 5742699/5773209 (99.5%)
residual std (active voxels): 2664.6387
Unsafe voxels (affected by a node connected to an inactive voxel): 508521/5773209 (8.8%)
Overlap-lost elements (lost material overlap or <50% material): 924/11088 (8.3%)

  cell e1 (max): mean=0.02341  std=0.00450  min=0.00801  max=0.03913
  cell e2 (mid): mean=-0.00612  std=0.00276  min=-0.01350  max=0.00582
  cell e3 (min): mean=-0.00941  std=0.00187  min=-0.01516  max=0.00351

Saved U_recovered.npy (22, 25, 23, 3), residual.npy (169, 193, 177),
      residual_std_cell.npy (21, 24, 22), zncc_cell.npy (21, 24, 22),
      principal_strains_cell.npy (21, 24, 22, 3), ...  -> test/
DVC COMPLETE
```

The finest scale converges at iteration 7 (`dU/U = 9.4×10⁻⁴ < eps = 1×10⁻³`)
with `std(res) ≈ 2663` grey-levels and 99.5 % active voxels.  The recovered
nodal displacements reflect the known ~19-voxel rigid shift (encoded in the
affine `t_x ≈ −19`), with small additional local deformation on top.

`write_outputs` also prints a `ZNCC (active voxels, no-ground-truth accuracy
proxy): ...` line right after the residual-std line above — a bounded
[-1, 1] correlation-quality figure (1 = perfect match) that needs no known
displacement field, so it's the metric to watch on real (non-ground-truth)
datasets like this one; see §9's sensitivity-sweep addendum for what it is
and is not good for.

Output files written to `test/`:

| File | Shape | Description |
|------|-------|-------------|
| `U_recovered.npy` | (22, 25, 23, 3) | Nodal displacements (full-image deformed coordinates) |
| `residual.npy` | (169, 193, 177) | Per-voxel final residual `f(x) − g(x+u)` |
| `principal_strains_cell.npy` | (21, 24, 22, 3) | Cell-centered principal strains in reference config |
| `principal_strains_cell_deformed.npy` | (21, 24, 22, 3) | Same, remapped to deformed config |
| `overlap_lost_element_mask.npy` | (21, 24, 22) | True where strain is unreliable (§7) |
| `unsafe_voxel_mask.npy` | (169, 193, 177) | True for voxels near the active-set boundary |
| `ref.npy` / `deformed.npy` | (169, 193, 177) | Cropped input volumes |
| `ref_forward_pushed.npy` | (169, 193, 177) | Reference warped forward by `U_local` |
| `run_log.txt` | — | Full stdout |

### Step 3 — Visualize

```bash
cd test && voxeldvc view-deformed
```

Opens a [napari](https://napari.org) viewer showing the three cell-centered
principal-strain components overlaid on `deformed.npy`.  The unsafe-voxel
mask is loaded as a second image layer so you can toggle it to see which
regions have reliable strain estimates.  Use the layer list on the left to
switch components; use the dimension slider to slice through the volume in
any axis.

## 4. The mesh / voxel / DOF relationship

This is the one piece of bookkeeping you need internalized before anything
else makes sense.

- You choose an **element edge length** `h` (in voxels) and the **element
  counts** `Nx_e, Ny_e, Nz_e` per axis.
- The image volume must then have shape `(Nx_e*h+1, Ny_e*h+1, Nz_e*h+1)` —
  one voxel layer of overlap between adjacent elements.
  `src/voxeldvc/engine/write_output.py` provides a helper to compute this
  automatically from an arbitrary image shape:

  ```python
  from voxeldvc.engine.write_output import mesh_dims
  Nx_e, Ny_e, Nz_e = mesh_dims(ref.shape, h)   # largest N per axis with N*h <= shape-1
  ```

  You don't need to enforce this by hand on real data: `voxeldvc preprocess`
  (§3, step 1) crops the overlap region to satisfy this constraint
  automatically for the `h` you pass it, snapping each axis down to the
  largest multiple of `h` that fits.

- **Nodes**: `(Nx_e+1) x (Ny_e+1) x (Nz_e+1)`, one per element corner. Node
  `(i,j,k)` sits at voxel position `(i*h, j*h, k*h)`.
- **DOFs**: `3*(Nx_e+1)*(Ny_e+1)*(Nz_e+1)`, node-major: `dof = 3*node_id + c`
  for displacement component `c in {0,1,2}` (ux,uy,uz). `U` arrays returned
  by the solver are flat 1D arrays in this ordering — to turn them back into
  a `(Nx+1,Ny+1,Nz+1,3)` grid use Fortran (`order='F'`) reshape per component:

  ```python
  def to_grid(comp, nbf):
      return comp.reshape(nbf, nbf, nbf, order='F')
  U_grid = np.stack([to_grid(U[c::3], Nx_e+1) for c in range(3)], axis=-1)
  ```
  (`voxeldvc.engine.write_output.write_outputs` does exactly this for you and
  saves the result as `U_recovered.npy`.)
- **Quadrature**: each element integrates over `(h+1)^3` points that coincide
  exactly with the voxel centers inside it (`geometry_dvc.VoxelCenterStencil`,
  `build_N_stencil`), with trapezoidal weights so a voxel shared by multiple
  elements still contributes total weight 1.
- There is no stored connectivity array — element/voxel/node indices are
  computed implicitly (`geometry_dvc.element_node_ids`,
  `kernels_dvc.get_elem_dofs`, `get_voxel_offsets`, `get_elem_voxel_origins`).

In 1D, with `h=2` (2 voxels per element) and 4 nodes (3 elements), the
picture looks like this — node `[A]` coincides spatially with the start of
voxel `1`, node `[B]` with the start of voxel `3`, and so on:

```
  A   B   C
┃1│2┃3│4┃5│6┃
```

A thick bar `┃` is a node (an element boundary); a thin bar `│` is just the
transition between two adjacent voxels inside an element, with no special
meaning of its own. Reading it: 4 nodes (the 4 thick bars) bound 3 elements
`A`, `B`, `C`, each spanning `h=2` voxels. Voxel numbering runs continuously
across the whole mesh (`1..6` here); the element labels are shown below the
bars, centered on the two voxels each element owns.

The same idea extends to 2D: element edges become lines, the intersections
of those lines are nodes, and voxels become dots scattered inside each
element (here a 2x2 grid of elements, `h=3`):

```
+-------+-------+
|  .  . |  .  . |
|       |       |
|  .  . |  .  . |
+-------+-------+
|  .  . |  .  . |
|       |       |
|  .  . |  .  . |
+-------+-------+
```

In 3D this same
`(Nx_e,Ny_e,Nz_e)` elements and edge length `h` has `(Nx_e*h+1, Ny_e*h+1,
Nz_e*h+1)` voxels and `(Nx_e+1, Ny_e+1, Nz_e+1)` nodes per the formulas
above — the 1D picture is literally that relationship along one axis, with
node `(i,j,k)` sitting at voxel position `(i*h, j*h, k*h)`.

## 5. Theory: how the DVC optimization problem is solved

This section explains the math behind `correlate_gpu`'s inner loop and why
the implementation never builds a global stiffness/Hessian matrix. It
assumes the mesh/voxel/DOF bookkeeping from §4.

### 5.1 The continuous problem

DVC seeks a displacement field `u(x)` such that the reference volume `f`
and the deformed volume `g` are related by the *brightness-conservation*
assumption

```
f(x) = g(x + u(x))      for all x in the overlap region.
```

This is a textbook ill-posed inverse problem: at a voxel with no local
intensity gradient (a flat/textureless region, or a pore), the equation
constrains nothing about `u` along directions perpendicular to `∇f`, so an
unregularized voxel-by-voxel solve is meaningless. Two things turn this into
a well-posed, solvable numerical problem:

1. **Finite-element discretization** reduces the infinite-dimensional `u(x)`
   to a finite set of nodal unknowns, with `u` interpolated *between* nodes
   by the Hex8 shape functions `N_a`:
   ```
   u_c(x) = sum_{a=0..7} N_a(x) * U[dof(e,a,c)]      (c = x,y,z component)
   ```
   for `x` inside element `e` (`geometry_dvc.element_node_ids` gives the
   element's 8 global node IDs; `dof = 3*node + c`). This alone regularizes
   the problem somewhat (one element's 8 nodes must explain `(h+1)^3` voxel
   constraints), but is not enough on its own for textureless/porous
   regions — hence point 2.
2. **Tikhonov-style regularization** (§5.4) adds a penalty term that
   prefers smooth (or mechanically admissible) displacement fields,
   resolving the remaining null space.

The discretized least-squares objective minimized at each Gauss-Newton step
is

```
J(U) = sum_q w_q * mask_q * [f(x_q) - g(x_q + u(x_q; U))]^2  +  ll * U^T R U
```

summed over every voxel-center quadrature point `q` of every element (with
trapezoidal weight `w_q` so a voxel shared by several elements still counts
once in total — see `geometry_dvc.VoxelCenterStencil`), `mask_q` excluding
no-material/out-of-overlap voxels, and `R` the regularization operator
(§5.4) weighted by `ll`.

### 5.2 Linearization: the Gauss-Newton normal equations

`g(x + u(x; U))` is nonlinear in `U` (it's `g` composed with the FE
interpolant), so `J(U)` is minimized iteratively. Given a current iterate
`U_k` with residual `res_q = f(x_q) - g(x_q + u(x_q; U_k))`, a Gauss-Newton
step looks for an increment `dU` linearizing `g(x_q + u(x_q; U_k + dU))` to
first order in `dU`:

```
g(x_q + u(x_q; U_k+dU)) ≈ g(x_q + u(x_q; U_k)) + ∇g(x_q + u(x_q; U_k)) · du(x_q; dU)
```

The classic Besnard-Hild-Roux/Lucas-Kanade-style DIC/DVC simplification
(used here, and matching `dic.Correlate`'s CPU reference implementation)
replaces `∇g` evaluated at the *deformed* position with `∇f` evaluated at
the *reference* position — valid to the same first order in `u`, and far
cheaper since `∇f` is computed once (`image_grad_and_values`, via
`xp.gradient`) and never recomputed inside the GN loop. With
`du_c(x_q) = sum_a N_a(x_q) dU[dof(e,a,c)]`, the chain rule gives the
per-quadrature-point sensitivity to each local DOF `(a,c)`:

```
d/dU[dof(e,a,c)]  g(x_q + u(x_q;U))  ≈  N_a(x_q) * ∇f(x_q)[c]
```

This 24-vector (8 nodes x 3 components) is exactly `kernels_dvc._v_eq`:

```
v_eq[q, 3*a+c] = N_stencil[q, a] * grad_f_vox[voxel(q)][c]
```

Substituting into the linearized least-squares residual
`res_q - v_eq(q)·dU_loc` and setting `dJ/d(dU) = 0` yields the Gauss-Newton
normal equations

```
(H + ll*R) dU = b - ll*R*U_k
```

with the *correlation Hessian* and right-hand side assembled by summing
rank-1 outer products / vectors over every quadrature point of every
element:

```
H   = sum_e sum_q  w_q * mask_q * v_eq(e,q) v_eq(e,q)^T
b   = sum_e sum_q  w_q * mask_q * res_q     * v_eq(e,q)
```

This is the Hessian of the Gauss-Newton-linearized least-squares cost — a
sum of outer products of a single 24-vector with itself, never a generic
dense or sparse matrix that needs explicit storage. That structure is what
makes the matrix-free implementation possible (§5.3).

### 5.3 Why no global matrix is ever assembled

A conventional FE/DIC code would assemble `H` into a global sparse matrix
(`Ndof x Ndof`, accumulating each element's local 24x24 contribution into
the right rows/columns) and factorize or iteratively solve the resulting
linear system. Two things make that approach unattractive here:

- `H`'s element-local contribution `v_eq v_eq^T` is *rank 1*, not a generic
  24x24 block — assembling and storing it as a 24x24 dense block per element
  (let alone a global `Ndof x Ndof` sparse matrix) wastes memory that scales
  with `Ne*24^2`, when the same information lives in the much smaller
  `(Ne*nq, 24)` array of `v_eq` vectors (`nq=(h+1)^3` per element).
- Real CT volumes push `Ndof` into the hundreds of thousands to low
  millions (`README.md`'s benchmark table: `Ndof` up to ~823,875 at `Nx=64`)
  with the per-voxel quadrature point count `nq` growing with `h^3` — a
  factorization-based direct solve, or even storing a sparse `H`, does not
  fit comfortably in GPU memory or scale well with mesh refinement.

Instead, `kernels_dvc.matvec_H` (and the regularizer matvecs
`matvec_K_ref`/`matvec_equilibrium_gap`) compute `H @ v` *on the fly*,
re-deriving each element's local contribution from `v` itself rather than
from a stored matrix, and the outer GN/PCG loop never needs anything but
this matvec. The pattern, shared by every matrix-free kernel in
`kernels_dvc.py`/`correlate_gpu.py`, is:

1. **Implicit connectivity, no stored arrays.** For a chunk of element IDs,
   `get_elem_dofs` derives each element's 24 global DOF indices by pure
   index arithmetic on the structured grid (`node = i + j*(Nx+1) +
   k*(Nx+1)*(Ny+1)`, `i,j,k` recovered from the element ID via `%`/`//`).
   `get_elem_voxel_origins`/`get_voxel_offsets` do the same for the
   element's `(h+1)^3` quadrature-point voxel indices. No connectivity
   table is ever stored — this is only possible because the mesh is a
   structured Hex8 grid with a fixed `h`, so every element's local
   numbering is a deterministic offset from a single "origin" index.
2. **Gather.** `v_loc = v[dofs]` and `gradf_q = grad_f_vox[vox_ids]` pull
   out exactly the slice of the global vector/field each element-chunk
   needs, batched as `(chunk, 24)` / `(chunk, nq, 3)` arrays.
3. **Local compute, vectorized over the chunk.** Per-quadrature-point
   sensitivities `v_eq` (`(chunk, nq, 24)`) and the rank-1 contraction
   `(v_eq · v_loc) * v_eq` are evaluated with batched `einsum` calls
   (`matvec_H`'s `'cqd,cd->cq'` then `'cq,cqd->cd'`) — this is the
   "matrix-free" step: `H`'s action on `v` is recomputed from `v_eq` for
   every matvec call, instead of reading a precomputed matrix entry.
4. **Scatter-add.** `xp.add.at(f, dofs.ravel(), Hv_loc.ravel())` accumulates
   each element's local result back into the global output vector. Because
   nodes (and voxels) are shared between adjacent elements, multiple chunks'
   contributions land on the same global index — `add.at` is the
   matrix-free analogue of a sparse assembly's "+=" into a shared row, and
   the trapezoidal `w_stencil` weights (§4) ensure a shared voxel's total
   weight across the elements that touch it is exactly 1, so this scatter-
   add reproduces the correct global sum without double counting.
5. **Chunking.** `Ne` elements are processed in chunks (`auto_chunk_size`
   picks a chunk size bounding the `(chunk, nq, 24)` tensor to ~20M
   entries) so memory scales with chunk size, not the full mesh — the same
   reason GPU memory use stays roughly constant as `Nx,Ny,Nz` grow (only the
   number of chunks increases).

The same five-step recipe assembles `b` (`correlate_gpu.compute_b_gpu`,
identical `v_eq` but contracted with the scalar residual `res_q` instead of
`dU`), the Jacobi preconditioner's diagonal (`build_diagonal_H`: same
contraction with `v_loc` replaced by reading off `v_eq^2` directly, since
`diag(v v^T) = v^2`), and both regularizer matvecs (§5.4) — there is exactly
one matrix-free "engine," reused for every linear operator the GN loop
needs.

### 5.4 Regularization, calibration, and the linear solve

`R` penalizes non-smooth or mechanically inadmissible displacement fields,
selectable via `reg_type`:

- **`'laplacian'`** (default): `R` is `build_K_ref_laplacian`'s 24x24
  reference "vector Laplacian" block, assembled matrix-free the same way as
  `H` via `matvec_K_ref` (`f = K @ v = sum_e E_elem[e] * P_e^T K_ref P_e
  v`, with `P_e` the implicit gather/scatter rather than a stored
  projector). Tikhonov/2nd-order smoothing of each displacement component
  independently.
- **`'equilibrium_gap'`**: a 4th-order, mechanically-motivated penalty
  `R_m = K_i^T K_i` (`K` the elastic Hex8 stiffness from `build_K_ref(nu)`,
  `K_i = P_i K` its restriction to interior DOFs via the diagonal 0/1
  projector `build_interior_dof_mask`). Since `K` is symmetric,
  `R_m @ v = K @ (P_i * (K @ v))` — two `matvec_K_ref` calls with an
  elementwise mask multiply in between (`matvec_equilibrium_gap`), so `R_m`
  is never assembled either, even as an intermediate. Its diagonal (needed
  for the Jacobi preconditioner) can't be read off element-local
  contributions the way `H`'s/`K`'s can, since `R_m` is itself a *product*
  of un-assembled operators — `build_diagonal_equilibrium_gap` instead uses
  a stochastic Hutchinson estimator (`E[z_i (R_m z)_i] ≈ diag(R_m)_i` over
  Rademacher probe vectors), trading exactness (not needed for a
  preconditioner) for staying fully matrix-free.

The regularization weight `ll` is calibrated, not hand-tuned per run: `ll =
c_reg * (l0/T)^reg_exponent * H0/L0`, where `l0` (voxels) is the user-chosen
regularization length, `T = 10*h` a reference length tied to the mesh, and
`H0/L0` a ratio of the correlation signal's "stiffness" to the
regularizer's, evaluated either via a plane-wave probe (`V.H.V / V.R.V`,
`reg_type='laplacian'`) or via `trace(H)/trace(R)` (`reg_type=
'equilibrium_gap'`, where the plane-wave ratio under-estimates `ll` by
several orders of magnitude for this 4th-order operator — see
`correlate_gpu`'s docstring). This keeps `ll` self-consistent across mesh
resolutions and pyramid levels (`multiscale_correlate_gpu` rescales `l0`
per level so the regularization length stays constant in physical units).

**Boundary effect — the `'laplacian'` regularizer drives the normal strain
toward zero in the outer element shell.** Despite the "vector Laplacian"
name, the *energy* this regularizer minimizes is the H¹ Dirichlet energy
`∫|∇u|²` — its `R = ∫ ∇N·∇N` block is exactly the operator whose quadratic
form `v^T R v` is that energy — i.e. a Tikhonov penalty on the displacement
gradient *itself*, weighted by `ll = c_reg*(l0/T)^2 * H0/L0` (`reg_exponent=2`;
e.g. `l0=36`, `T=10*h=80` give `(36/80)^2 ≈ 0.20 * H0/L0`). Minimizing
`∫|∇u|²` subject to the image data imposes the *natural* (Neumann) boundary
condition `∂u/∂n = 0` on the domain faces, so the recovered **normal strain
is pulled toward zero in the one-element-deep boundary layer**. The bias is
proportional to the true normal strain on each face, so it is only
conspicuous where that strain is large. On a uniform synthetic field
`eps_zz = -1.0e-2`, `eps_xx = eps_yy ≈ +2.6e-3` (element size `h=8`), the
interior recovered `e3` (most-compressive principal strain) sits at
`≈ -9.6e-3` — essentially the imposed value — but the two z-faces collapse
to `≈ -4.8e-3` and `≈ -0.7e-3`, while the x/y faces (true strain ~4× smaller)
stay visibly flat. This is inherent to the H¹ penalty, not a solver defect
or a loss of image overlap (the boundary elements remain fully correlated).
Because of this bias, the outer one-element shell is **always flagged as
untrustworthy** by the validity masks of §7 — `overlap_lost_element_mask`
(the strain gate) and `unsafe_voxel_mask`/`unsafe_element_mask`/
`unsafe_node_mask` all union in `boundary_element_mask(Nx,Ny,Nz)` as a
standing criterion, independent of the active/material masks, so a consumer
that respects those masks drops the shell automatically. (On a small mesh the
shell is a large fraction — e.g. 218/343 elements on a 7³ mesh — but on a
production-sized mesh it is a thin rind.) Further reduce the residual bias by
lowering `l0` (weaker penalty, at the cost of more noise — see §9's `h`/`l0`
sweep), or crop less in preprocessing so the region of interest sits further
into the mesh interior. The `'equilibrium_gap'`
regularizer penalizes only *interior*-node equilibrium and so avoids this
boundary bias in principle, but it is far less robust to a cold
(≈identity-prealign) start under `dynamic_mask=True` and can diverge outright
unless warm-started from a converged `'laplacian'` solution.

Finally, `(H + ll*R) dU = b - ll*R*U_k` is solved by **Jacobi-preconditioned
Conjugate Gradient** (`kernels_dvc.pcg`) rather than direct factorization —
the only solver compatible with a matrix that is never assembled: `pcg`
needs nothing but a `matvec` callable (`v -> (H+ll*R)@v`, itself built from
the matrix-free kernels above) and the operator's diagonal as a Jacobi
preconditioner (`build_diagonal_H`/`diag_reg`, summed since `diag(A+B) =
diag(A)+diag(B)`). The outer Gauss-Newton loop in `correlate_gpu` repeats
this linearize-and-solve step (recomputing `res`, the dynamic in-bounds
mask, and — if `dynamic_mask=True` — `H` itself, every iteration) until
`||dU||/||U|| < eps`. See `CLAUDE.md`'s "PA6GF30 CT results" entry on
`freeze_mask_after` for the practical caveats around mask freezing.

### 5.5 Multiscale pyramid and affine prealignment

`multiscale_correlate_gpu(scales=(2,1,0), ...)` runs the Gauss-Newton solve
above through a coarse-to-fine image pyramid: `f`/`g` are decimated by
`2**iscale`, the element size shrinks to `h_i = h // 2**iscale`, and the
recovered `U` is carried forward as the next level's initial guess `U0`.
This matters because a zero-initialized coarsest level can fail to resolve
a displacement that's large relative to its (coarse) mesh.

Two ways to seed the coarsest level with something better than zero:

- **`fft_prealign=True`** (default): before the pyramid loop, an FFT
  phase-correlation (`fft_rigid_shift`) estimates a rigid shift and uses it
  as `U0` for the coarsest scale.
- **`prealign_affine=True`** (takes precedence over `fft_prealign`): runs
  `affine_prealign` once on the full-resolution images — a 9-DOF
  Gauss-Newton/Powell search over rotation, independent per-axis scale, and
  translation, minimizing 1-ZNCC on a `decimate=4` subsample with
  no-material voxels (intensity `<= glt`, the `--glt` threshold; default
  `0`) excluded on both sides — and converts
  the recovered affine `(A, t, c)` into a uniform `U0`
  (`u(x) = A @ (x - c) + c + t - x`, rescaled per pyramid level).

On the bundled PA6GF30 dataset, `prealign_affine` recovered a ~1-3%
per-axis scale correction (negligible rotation) and converged to the same
final solution as `fft_prealign` alone but in ~16% fewer total GN
iterations (the `2^1` scale met `eps=1e-3` in 24 iters instead of running
the full 30 unconverged, and `2^0` converged in 4 iters instead of 9).
Feeding the *already-decimated* coarsest-level images into
`affine_prealign` (instead of the full-resolution ones) was tried first and
performed markedly worse — at that resolution the 9-DOF search lands in a
poor local minimum and bakes a spurious rotation/scale into `U0`, producing
large strain outliers and a 42% higher final residual.

### 5.6 GPU memory footprint and how it is kept small

**Rule of thumb: the solve needs ~49 MB of GPU VRAM per million reference
voxels** (`float32`, `h=8`, `scales=(2,1,0)`). Measured with `nvidia-smi`:
**10920 MB for a `(1100, 450, 450)` reference** (≈222.75 M voxels). So the
largest cubic volume that fits is roughly `cube-root(VRAM_MB / 49 × 1e6)`:

| GPU | VRAM | largest cube (headroom) | absolute |
|-----|------|-------------------------|----------|
| L40S | 48 GB | ~**965³** | ~1000³ |
| RTX 4090 / A6000 | 24 GB | ~765³ | ~795³ |
| RTX 4060 Ti | 8 GB | ~520³ | ~545³ |

Leave ~10 % headroom for the CUDA context and memory-pool fragmentation.
Two things make the real number a little higher than the reference-voxel
count suggests: the **deformed** image is cropped to the (usually slightly
larger) affine bounding box, and a large rotation/strain inflates that box
further.

**Why it costs what it does, and the key idea.** The footprint is dominated
by *per-voxel and per-quadrature-point* arrays — the reference gradient
`grad_f_vox` (shape `(Nvox, 3)`), the deformed image, the residual, etc. —
all sized by the voxel count `Nvox = (Nx·h+1)(Ny·h+1)(Nz·h+1)`, **not** by
the element or DOF count. In a multiscale run `Nvox` grows ~8× per finer
pyramid level (`h_i` doubles), so the finest level dominates.

The central technique for keeping VRAM small is the same matrix-free
**element chunking** used for the Hessian/RHS assembly (§5.3), applied to
*every* stage that would otherwise touch a full-resolution buffer:

- **Deformed-image sampling** is done in element chunks
  (`sample_deformed_field`): the deformed sample coordinates `x + u(x)`, the
  in-bounds/`g>0` mask, and the interpolated `g` value are computed for one
  `(chunk, (h+1)³)` tile at a time and scattered into the output, so the full
  `(3, Nvox)` coordinate array — the single largest transient — is never
  built. This alone roughly **halved** the peak.
- **Normalization statistics** (the ZNCC mean/std) are accumulated by a
  chunked reduction (`masked_quadpoint_mean_std`) rather than gathering the
  full `Ne·(h+1)³` quad-point array.
- **Setup transients** are minimized (the reference gradient is written one
  axis at a time instead of a `stack` of three ravel copies) and the CuPy
  memory pool is released once after setup, before the Gauss-Newton loop.

These changes are numerically transparent (the sampling ones are
bit-identical; the reductions match to ~1e-8, below the GPU's own
run-to-run atomic-add noise) and took the same `(1100,450,450)` case from
**30608 MB → 10920 MB** (137 → 49 MB/Mvox). See `CLAUDE.md`
(§ *GPU VRAM requirements*) for the full commit-by-commit breakdown.

**Practical knobs if you are VRAM-limited:** reduce `h` (fewer quadrature
voxels per element — the strongest lever, since `Nvox ∝ h³`), keep the
images `uint16`/`float32` rather than `float64` on disk, or drop the finest
pyramid scale. The int32 index guard in `correlate_gpu` caps any single
level at `Nvox < 2.1×10⁹` (cube side ~1290), but VRAM is the binding limit
well before that on current cards.

## 6. Output files

`voxeldvc run` calls `voxeldvc.engine.write_output.write_outputs` after the
solve and writes the following files to the work directory:

Files written to `<work_dir>/`:

| File | Shape | Field type | Meaning |
|---|---|---|---|
| `U_recovered.npy` | `(Nx+1,Ny+1,Nz+1,3)` | nodal | recovered displacement |
| `residual.npy` | `(Nvx,Nvy,Nvz)` | voxel | final per-voxel residual |
| `ref.npy` | `(Nvx,Nvy,Nvz)` | voxel | reference image (as passed in) |
| `deformed.npy` | `(Nvx,Nvy,Nvz)` | voxel | deformed image, cropped to ref's region |
| `principal_strains_cell.npy` | `(Nx,Ny,Nz,3)` | cell | principal small strains `e1>=e2>=e3` at element centroids, reference config |
| `principal_strains_cell_deformed.npy` | `(Nx,Ny,Nz,3)` | cell | same, remapped to deformed config (griddata) |
| `equivalent_strain_cell.npy` | `(Nx,Ny,Nz)` | cell | von Mises equivalent (deviatoric) strain `eps_eq` per element, reference config — a one-channel summary of the three principal strains |
| `equivalent_strain_cell_deformed.npy` | `(Nx,Ny,Nz)` | cell | same, remapped to deformed config |
| `residual_std_cell.npy` | `(Nx,Ny,Nz)` | cell | per-element residual std (grey levels), over the element's active voxels; `NaN` where too few active voxels |
| `zncc_cell.npy` | `(Nx,Ny,Nz)` | cell | per-element ZNCC between `f` and `g(x+u)` over active voxels, in `[-1,1]` (1 = perfect local correlation); `NaN` where degenerate — the per-cell analog of the global ZNCC proxy |
| `gn_convergence_cell.npy` | `(Nx,Ny,Nz)` | cell | per-element final Gauss-Newton `dU/U` (only written when the solver supplies it) |
| `overlap_lost_element_mask.npy` | `(Nx,Ny,Nz)` | cell | `True` = cell-strain not trustworthy (see §7) |
| `overlap_lost_element_mask_deformed.npy` | `(Nx,Ny,Nz)` | cell | same mask, remapped to deformed config |
| `material_cell_mask.npy` | `(Nx,Ny,Nz)` | cell | `True` where the element's *mean* reference gray level exceeds `glt` (a material/void gate on the strain fields) |
| `material_cell_mask_eroded.npy` | `(Nx,Ny,Nz)` | cell | `material_cell_mask` eroded by one element layer (drops elements bordering pores/background), for gating strains away from the material/void interface |
| `unsafe_voxel_mask.npy` | `(Nvx,Nvy,Nvz)` | voxel | `True` = voxel touches an element with any inactive voxel (1-element dilation of lost overlap, treats porosity as a defect) |
| `unsafe_voxel_mask_forward_pushed.npy` | `(Nvx,Nvy,Nvz)` | voxel | `unsafe_voxel_mask` pushed forward into deformed coordinates |
| `ref_forward_pushed.npy` | `(Nvx,Nvy,Nvz)` | voxel | reference image warped forward by `U` (`ref(x - u(x))`), i.e. a prediction of the deformed image — overlay against `deformed.npy` as a visual sanity check |
| `dvc_fields.vti` | — | mesh grid | ParaView/VisIt-ready binary XML ImageData: nodal displacement as POINT data plus the principal + equivalent strains and the per-cell quality/material fields (`material_mask`, `material_mask_eroded`, `safe_elements`, `residual_std`, `zncc`, and `gn_convergence` when available) as CELL data, on one `spacing=(h,h,h)` grid. Written through a guarded `vtk_export` call, so a missing/broken vtk backend only skips this file — the `.npy` outputs are unaffected |

Full stdout is also saved to `run_log.txt` in the work directory.

## 7. Two flavors of "this region isn't trustworthy"

The codebase has two different masks answering "where can I trust the
output," and they answer different questions — don't conflate them.
Both unconditionally flag the outer one-element boundary shell
(`boundary_element_mask`) as a standing criterion, on top of the
mask-specific criteria below, because that shell's normal strain is biased
by the regularizer's Neumann natural BC regardless of image content (§5.4):

- **`unsafe_voxel_mask` / `compute_unsafe_element_mask` /
  `compute_unsafe_node_mask`** (`src/voxeldvc/engine/kernels_dvc.py`): flag
  anything touching an element with *any* inactive voxel, full stop —
  including normal pores in a porous material. On a porous specimen like
  PA6GF30 this flags the large majority of voxels, which is correct but not
  useful for judging *strain* validity in a material that's naturally
  porous.
- **`compute_overlap_lost_element_mask`** (also `kernels_dvc.py`): the
  cell-strain-specific gate, porosity-aware. An element is flagged `True`
  only if (1) its *material* (`mask0 = ref>glt`, the `--glt` threshold) lost
  correlation overlap — the deformed position is out of bounds or lands on
  `g<=glt` — dilated by one
  element through shared nodes, OR (2) its material count fraction (#
  `mask0==True` voxels / `(h+1)^3`) is below `min_material_frac` (default
  0.5), i.e. it's mostly pore so its strain is regularization-governed
  rather than measured. On a fully dense specimen this reduces to criterion
  1 plus the boundary shell (verified against the porosity-free elastic
  ground-truth dataset, see `CLAUDE.md`'s `compute_overlap_lost_element_mask`
  note). Use this mask, not `unsafe_voxel_mask`, to gate
  `principal_strains_cell.npy`.

## 8. Visualizing results (napari)

All `voxeldvc view-*` subcommands load their inputs by bare relative `.npy`
filename, so **always run them from inside the relevant output directory**:

```bash
cd dvc_PA6GF30 && voxeldvc view-displacements
```

Every script follows the same napari pattern: load a multi-component field
with `napari.imshow(data, channel_axis=3, ...)` (one layer per component,
`turbo` colormap, colorbar on, only the first component visible by default),
then overlay one or two supporting images/masks at reduced opacity so you can
toggle them from the layer list. `viewer.dims.order = (1, 2, 0)` puts the
slice-through axis last so the default 2D slice is the XY plane.

| Subcommand | Run from (output dir of) | Loads | Shows |
|---|---|---|---|
| `voxeldvc view-displacements` | `voxeldvc run` (incl. `voxeldvc ground-truth`) | `U_recovered.npy`, `ref.npy`, `residual.npy`, `unsafe_voxel_mask.npy` | recovered nodal `ux,uy,uz` over the reference image, with residual and unsafe-voxel overlays |
| `voxeldvc view-displacements-gt-compare` | `voxeldvc ground-truth` only | `U_recovered_vox.npy`, `U_gt_vox.npy`, `U_error_vox.npy`, `ref.npy`, `voxel_recoverable_mask.npy` | per-voxel recovered vs. prescribed (ground-truth) displacement and their difference — 9 component layers (`ux/uy/uz` x recovered/gt/error) plus the recoverable-voxel mask; see §9 |
| `voxeldvc view-reference` | `voxeldvc run` (incl. `voxeldvc ground-truth`) | `equivalent_strain_cell.npy`, `principal_strains_cell.npy`, `ref.npy`, `unsafe_voxel_mask.npy` | reference-config equivalent strain `eps_eq` and principal strains `eps1,eps2,eps3` over the reference image |
| `voxeldvc view-deformed` | `voxeldvc run` (incl. `voxeldvc ground-truth`) | `equivalent_strain_cell_deformed.npy`, `principal_strains_cell_deformed.npy`, `deformed.npy`, `unsafe_voxel_mask_forward_pushed.npy` | same equivalent + principal strains remapped to the deformed config, over the deformed image |
| `voxeldvc view-overlay` | `voxeldvc preprocess` | `ref_preprocessed.npy`, `def_preprocessed.npy` | additive pink (reference) / green (deformed) overlay — white/gray where they align — for a quick visual sanity check of the crop/alignment *before* running the solver |

`view_displacements.py`'s pattern (`voxeldvc view-displacements`), reproduced
in full since the others differ only in which fields they load:

```python
import napari
import numpy as np

lc = 8  # h, for physical-unit scaling
scale = [1/lc] * 3

data = np.load('U_recovered.npy')                       # (Nx+1,Ny+1,Nz+1,3)
viewer, image_layers = napari.imshow(
    data, channel_axis=3, name=["ux", "uy", "uz"],
    colormap=["turbo", "turbo", "turbo"])
for layer in image_layers:
    layer.colorbar.visible = True
    layer.visible = False
image_layers[0].visible = True

im0 = np.load('ref.npy')
viewer.add_image(im0, scale=scale, opacity=0.5)

residuals = np.load('residual.npy')
viewer.add_image(residuals, scale=scale, opacity=0.5)

unsafe = ~np.load('unsafe_voxel_mask.npy')
viewer.add_image(unsafe, scale=scale, opacity=5)

napari.run()
```

`voxeldvc view-reference` and `voxeldvc view-deformed` follow the same shape,
swapping in strain fields (`principal_strains_cell.npy`) or the
deformed/forward-pushed volumes instead.

`voxeldvc view-displacements-gt-compare` follows the same shape three times
over — one `napari.imshow(..., channel_axis=3)` call each for the recovered,
ground-truth, and error voxel fields, added to the *same* viewer via
`napari.imshow(..., viewer=viewer)` — plus `ref.npy` for spatial context and
`~voxel_recoverable_mask.npy` (voxels whose ground-truth-deformed position
falls outside the deformed image, so their error reflects missing data, not
DVC accuracy — see §9). Only `ux_recovered`/`ux_gt`/`ux_error` are visible by
default; toggle the other six component layers from the layer list.
`voxeldvc view-overlay` is the odd one out: it runs on `voxeldvc preprocess`'s
raw output (not a DVC result) and uses `blending='additive'` with `magenta`/
`green` colormaps instead of the `channel_axis` pattern, so reference and
deformed volumes are overlaid directly with per-image percentile-based
contrast limits rather than shown as separate toggleable layers.

## 9. Validating against ground truth

`voxeldvc ground-truth` (`src/voxeldvc/run_ground_truth_pipeline.py`) is the
canonical end-to-end accuracy check — worth reading in full if you change
anything in the GN loop or regularization. Unlike the rest of this section's
scripts, it doesn't run the correlation itself: it chains 4 stages, each a
separate subprocess (`python -m voxeldvc.<module>`), through the exact same
`voxeldvc preprocess` → `voxeldvc run` path used on real
(unknown-ground-truth) data, so this check exercises the affine
pre-alignment / overlap-crop / `ext_affine` machinery too, not just the GN
solver:

1. **`voxeldvc prepare-ground-truth`** loads
   `assets/GT_z+0.01/ref.npy`
   (65³ uint16 reference) and `assets/GT_z+0.01/U.npy`
   (66³, 3-channel ground-truth displacement on integer coordinates 0..65),
   interpolates the GT field onto the 65³ image grid, and generates the
   deformed image by backward-warping the reference:
   `deformed(x) = ref(x - u_gt(x))` (`RegularGridInterpolator`, trilinear,
   `fill_value=None` for extrapolation at the boundary). Writes `ref.npy`,
   `deformed.npy` and `u_gt_vox.npy` (the interpolated GT field, at every
   reference voxel) to `<output-dir>/gt_input/`.
2. **`voxeldvc preprocess`** (§10) runs unmodified on that pair: affine
   pre-alignment + overlap crop, writing into `<output-dir>/`.
3. **`voxeldvc run`** (§10) runs unmodified on the preprocessed crop,
   writing `U_recovered.npy` and the rest of the standard output set (§8's
   table) into `<output-dir>/`.
4. **`voxeldvc analyse-ground-truth`** compares `U_recovered.npy` against
   `u_gt_vox.npy`, undoing the two coordinate-frame offsets preprocessing
   introduces (crop origin on the reference side, `def_lo` on the deformed
   side — see the script's module docstring) so the comparison is
   apples-to-apples regardless of where `voxeldvc preprocess` cropped to:
   - **At mesh nodes**: per-component mean/std/RMS/max error, reported over
     three regions: **recoverable** (full voxel support inside the deformed
     image), **valid**, and **truncated** (some support voxels map out of
     bounds under the GT field, so the DVC literally cannot see that data).
     `truncated`-node error reflects missing data, not method failure, so it
     is excluded from both accuracy metrics. `valid` additionally drops the
     nodes the pipeline itself flags untrustworthy on real data — chiefly the
     regularizer-biased outer boundary shell (§5.4/§7), loaded from the saved
     `overlap_lost_element_mask.npy` — and is therefore the **fair measure of
     method accuracy**; `recoverable` still over-counts the boundary shell
     (on the bundled dataset it roughly *doubles* the reported RMS). The
     analogous solver masks gate the corresponding real-data output, so the
     valid metric scores exactly what a real run would trust.
   - **At every voxel** of the reference crop: `U_recovered` is evaluated at
     each voxel center via `voxeldvc.engine.correlate_gpu.interp_field_vox` —
     the same trilinear FE shape-function evaluation the solver itself
     uses — and compared component-wise against the ground truth at voxel
     resolution. Two tables are printed: over **recoverable** voxels (GT-
     deformed position in bounds) and over **valid** voxels (also dropping
     `unsafe_voxel_mask.npy`, i.e. the boundary shell + overlap-lost voxels).
     If the solver mask files are absent (a `work_dir` predating them), the
     `valid` rows are silently skipped.

Run it with `voxeldvc ground-truth --output-dir DIR`. The GT-source flags
default to the bundled tension dataset, so the shortest invocation is:
```
voxeldvc ground-truth --output-dir ground_truth
```
which is equivalent to spelling the defaults out:
```
voxeldvc ground-truth --gt-ref-file assets/GT_z+0.01/ref.npy \
                      --gt-displacement-file assets/GT_z+0.01/U.npy \
                      --output-dir ground_truth
```
A second bundled dataset imposes a ~5% uniaxial compression along z:
```
voxeldvc ground-truth --gt-ref-file assets/GT_z-0.05/ref.npy \
                      --gt-displacement-file assets/GT_z-0.05/U.npy \
                      --output-dir gt
```

 (`--output-dir` is
required, no default); outputs land in `DIR/` relative to the current
directory
(`gt_input/` for stage 1; stages 2-4's outputs, including `run_log.txt`,
`ground_truth_comparison_log.txt`, `U_gt_nodes.npy`, `U_error.npy`,
`node_truncated_mask.npy`, `node_oob_fraction.npy`, `node_valid_mask.npy`,
and the per-voxel `U_gt_vox.npy`, `U_recovered_vox.npy`, `U_error_vox.npy`,
`voxel_recoverable_mask.npy`, `voxel_valid_mask.npy`, share the root
directory). Visualize the
per-voxel comparison with `voxeldvc view-displacements-gt-compare` (§8). Since
the mesh isn't guaranteed to stay cubic or full-image-sized (the overlap crop
can clip axes differently), the comparison scripts read `Nx_e, Ny_e, Nz_e`
from the actual output shapes rather than assuming `8,8,8`.

Each stage can also be run standalone via its own subcommand (e.g.
`voxeldvc prepare-ground-truth` to inspect `gt_input/` before preprocessing,
or `voxeldvc analyse-ground-truth` to re-run just the comparison after
tweaking it) — see each subcommand's `--help`.

### Imposing a synthetic strain instead of the CT ground truth

`--tensile-strain STRAIN` (forwarded to `voxeldvc prepare-ground-truth`)
replaces the loaded CT displacement field with a synthetic uniaxial field
`u_x(x,y,z) = STRAIN * x`, `u_y = u_z = 0` (same `(66,66,66,3)` layout as the
loaded field, so everything downstream — warping, preprocessing, DVC, both
comparisons — is unchanged). Positive `STRAIN` is tension, negative is
compression:

```bash
voxeldvc ground-truth --output-dir gt_out --tensile-strain 0.01    # +1% uniaxial tension
voxeldvc ground-truth --output-dir gt_out --tensile-strain -0.02   # -2% uniaxial compression
```

This is a useful known-answer check independent of the bundled CT dataset's
specific (and somewhat irregular) displacement field: `du_x/dx` should come
back close to `STRAIN` with `uy`/`uz` near zero. A couple of things to expect:

- **Tension truncates boundary nodes/voxels; compression doesn't.** Tension
  stretches material toward and past the image's far `x` edge, so some node/
  voxel support maps out of the deformed image (`truncated`/`unrecoverable` —
  see stage 4 above); compression pulls material inward, so at moderate
  magnitudes every node stays fully recoverable.
- **Error grows with `|STRAIN|`.** E.g. at `h=8` on this dataset, `+1%` gives
  a recoverable-node RMS error around 0.015 voxels — roughly proportional to
  the strain magnitude, as expected for a Gauss-Newton correlation whose
  linearization error scales with the deformation size.

### Sensitivity sweep over h and l0

`voxeldvc ground-truth --h H --l0-factor L0_FACTOR` (forwarded from
`--l0-factor` down to `voxeldvc run`'s `l0 = L0_FACTOR * h`, default `4.5`
matching the previous hardcoded value) makes both mesh size and
regularization length sweepable independently, so this pipeline doubles as
an accuracy-sensitivity harness for the two knobs called out in §5.4/§11. A
35-run sweep (h ∈ {4, 8, 12, 16, 20} voxels × l0_factor ∈
{1, 2, 3, 4.5, 6, 9, 12}, scored by *recoverable*-node RMS error — this sweep
predates the `valid` metric, which is now the fairer measure) found:

- Every `h` traces the same U-shaped accuracy curve in `l0`: too little
  regularization under-constrains the per-element solve; too much
  over-smooths the real field.
- The RMS-minimizing `l0_factor` decreases as `h` grows, but the minimizing
  *absolute* `l0` clusters around 20–36 voxels at every tested `h` — `l0`
  behaves as a roughly mesh-resolution-independent physical smoothing
  length, matching `multiscale_correlate_gpu`'s per-level rescaling intent,
  rather than a fixed fraction of `h`.
- `h=4` is fragile: `l0_factor=1–2` (`l0=4–8` voxels) makes the GN solve
  diverge outright (RMS 2.6–7.8 voxels, `dU/U` never drops below `eps`) — a
  fine mesh has too few quadrature points per element to stay well-posed
  without enough regularization.
- The codebase-wide default (`h=8`, `l0_factor=4.5` → `l0=36`) scores 0.0175
  voxels *recoverable* RMS, close to the sweep's best of 0.0159 voxels (`h=4`,
  `l0_factor=9`) and 0.0163 voxels (`h=16`, `l0_factor=2`) while avoiding
  `h=4`'s divergence cliff, so it remains a reasonable default. Its **valid**
  RMS (fair measure, boundary shell excluded) is **0.0060 voxels** — the
  boundary shell roughly halves the headline recoverable error, and the valid
  metric keeps the same U-shape (measured at `h=8`: `l0_factor=4.5` → valid
  0.0060 vs. `l0_factor=9` → valid 0.0102).
- Caveat: recoverable-node counts shrink sharply with `h` (3611 at `h=4` down
  to ~29–30 at `h=16`/`h=20`), so the coarse-mesh rows are noisier estimates;
  the `valid` set is smaller still, since it drops the outer element shell.

Full heatmap, per-`h` line chart, and raw data table:
https://claude.ai/code/artifact/0e21a2e4-3183-4248-b46d-f192d32bd1cc

`voxeldvc sweep-ground-truth` (`src/voxeldvc/ground_truth/sweep_ground_truth.py`)
automates this: it drives `voxeldvc ground-truth` once per `--h`/`--l0-factor`
combination (comma-separated lists, default `4,8,16` × `2.25,4.5,9`), each
into its own `<output-root>/h<h>_l0f<l0_factor>/` subdirectory, then reads
back **both** the recoverable- and valid-node RMS/max error from that combo's
`U_error.npy` + `node_truncated_mask.npy`/`node_valid_mask.npy` and the final
`dU/U` from `run_log.txt`, printing a summary table (with both metrics and a
"best" row for each; optionally also written to CSV with `--csv PATH`). A
combo whose recoverable RMS exceeds 1 voxel is flagged `DIVERGED`.
Unlike `voxeldvc prepare-ground-truth`/`voxeldvc ground-truth` (which fall
back to the `assets/GT_z+0.01` dataset when its GT-source flags are omitted),
`sweep-ground-truth` requires `--gt-ref-file` and `--output-root` and one of
`--gt-displacement-file`/`--tensile-strain` on every invocation — a 9+ run
sweep is expensive enough that silently defaulting to the bundled dataset on
a typo'd flag would be a costly mistake to discover after the fact:

```bash
voxeldvc sweep-ground-truth \
  --gt-ref-file assets/GT_z+0.01/ref.npy \
  --gt-displacement-file assets/GT_z+0.01/U.npy \
  --output-root dvc_elastic_groundtruth_sweep \
  --h 4,8,16 --l0-factor 2.25,4.5,9 --csv sweep.csv
```

`--gt-ref-file PATH` and `--gt-displacement-file PATH` point at a
known-displacement dataset (forwarded through to every combo's
`prepare-ground-truth` stage). `--gt-ref-file` is a reference image, shape
`(N,N,N)`; `--gt-displacement-file` is the matching ground-truth
displacement, shape `(N+1,N+1,N+1,3)` — the displacement grid has **one more
point per axis than the image**: it's sampled at integer coordinates `0..N`
while the image's voxels sit at `0..N-1`, so interpolating the displacement
field down onto every image voxel never needs extrapolation. `voxeldvc
prepare-ground-truth`/`voxeldvc ground-truth`/`voxeldvc sweep-ground-truth`
all validate this relationship up front and raise a clear error naming both
shapes if it doesn't hold. `--gt-displacement-file` and `--tensile-strain`
are mutually exclusive and exactly one is required.

**Auto-tuning `h`/`l0_factor` for your own dataset**: `--gt-ref-file` and
`--tensile-strain` combine to turn this into a per-dataset parameter search
that needs nothing but a reference image — no CT ground-truth displacement
field required, since `--tensile-strain` synthesizes one. Point
`--gt-ref-file` at your own reference image (shape `(N,N,N)`) and impose a
known synthetic strain with `--tensile-strain`:

```bash
voxeldvc sweep-ground-truth --gt-ref-file sweep_input/original_intensity.npy \
  --tensile-strain 0.01 --output-root sweep_out
```

(`0.01` is just an example strain magnitude — any value in the small-strain
regime works, per the caveat in "Imposing a synthetic strain instead of the
CT ground truth" above.) `sweep-ground-truth` runs every `--h`/`--l0-factor`
combination against this known-answer deformation of your own image and
prints, at the end of its summary table, the best combination by **both**
the recoverable- and valid-node RMS ("Best by recoverable RMS" / "Best by
valid RMS (fair measure)" lines) — now tuned to your dataset's actual image
content instead of the bundled ground-truth dataset's. Prefer the valid-RMS pick.

### Picking l0 when there is no ground truth: ZNCC, and why it's not enough alone

Every real dataset lacks the known displacement field this sweep used to
score accuracy. `voxeldvc.engine.write_output.compute_zncc` fills that gap
with a metric that needs none: the **zero-normalized cross-correlation**
between the reference image and the deformed image warped by the recovered
`U`, over the active/overlap voxel set — bounded [-1, 1] (1 = perfect match),
comparable across meshes/masks (unlike raw residual std, which is in grey
levels and biased by the active-voxel count). `voxeldvc run` now prints
it on every run (`ZNCC (active voxels, no-ground-truth accuracy proxy):
...`, right after the residual-std line in §3's example output).

Re-running the same 35-point h/l0 sweep scored by ZNCC instead of the
ground-truth RMS found:

- **ZNCC reliably catches outright divergence.** The two h=4 configs that
  diverged above (`l0_factor=1, 2`) collapse ZNCC to 0.13 and 0.38, vs.
  ~0.93–0.94 everywhere else — an unambiguous, no-ground-truth-needed
  signal that something is badly wrong.
- **But within the well-posed regime, ZNCC is monotonic in `l0`** — it
  always prefers *less* regularization and never reproduces the true
  U-shaped accuracy curve from the ground-truth sweep. Maximizing ZNCC
  alone systematically walks toward under-regularization (raw residual std
  shows the identical bias, for the same reason).
- **The resulting mismatch grows on finer meshes**: picking `l0` by ZNCC
  alone gives RMS error 2.41× the true optimum at `h=4`, 1.26× at `h=8`,
  1.29× at `h=12`, 1.09× at `h=16`, and matches exactly at `h=20` (the
  coarsest mesh tested, where the element itself already does most of the
  smoothing so minimal extra regularization happens to be correct either
  way).

**Practical takeaway**: use ZNCC (or residual std) as a divergence/sanity
check — flag failed runs, or compare different preprocessing/affine
choices — but don't tune regularization strength by maximizing it alone. A
held-out-voxel or L-curve method (trading the data-residual term against
the regularization-term norm as `l0` varies) would be the principled
no-ground-truth way to pick `l0`.

Full ZNCC heatmap, line-chart comparison, and raw data:
https://claude.ai/code/artifact/c46f8d73-e3aa-4062-8f51-72b9317421ff

## 10. Other scripts at a glance

- `voxeldvc preprocess <ref> <def> [--h H] [--glt GLT] [--bin B]
  [--decimate D] [--no-affine-refine] [--crop X Y Z] [--output-dir DIR]`:
  affine pre-alignment (coarse ZNCC/Powell prealign + full-resolution
  Gauss-Newton refinement) + overlap crop; `--glt` sets the gray-level
  threshold below which a voxel is treated as no material / no contrast
  (default 0, saved to the JSON and reused by `run`); `--decimate` is the
  downsampling factor for the coarse prealign (default 4; 1 disables it);
  `--crop X Y Z` symmetrically crops that many voxels off each axis before
  alignment (default `0 0 0`); writes
  `ref_preprocessed.npy`, `def_preprocessed.npy`, `def_ref_aligned.npy`, and
  `affine_prealign.json` into the output directory.
- `voxeldvc run <work_dir> [--l0-factor L0_FACTOR]`: full multiscale DVC on
  a preprocessed work directory; reads the three `.npy` files and JSON,
  writes all output fields. `l0 = L0_FACTOR * h` (default 4.5).
- `voxeldvc prepare-ground-truth [--tensile-strain STRAIN] [--output-dir DIR]` /
  `voxeldvc analyse-ground-truth [--work-dir DIR] [--gt-dir DIR]`: stages 1
  and 4 of the ground-truth pipeline (§9) — generate the known-displacement image
  pair, and compare a `voxeldvc run` result back against it. Normally run via
  `voxeldvc ground-truth`, not directly.
- `voxeldvc compute-residuals <work_dir>`: standalone post-processing that
  recomputes the ZNCC-normalized per-voxel DVC residual from a finished
  `voxeldvc run` output directory and reports the same `std(res)` (grey levels)
  the GN loop prints — useful for re-checking a result without re-running the
  solve.
- `voxeldvc crop-array <input.npy> <output.npy> Nx Ny Nz [--offset OX OY OZ]`:
  crop a saved `.npy` (or `.tif`) volume to `Nx×Ny×Nz`, centered with an
  optional offset.
- `voxeldvc mhd-to-npy <file.mhd> [...] [-o DIR] [--histogram] [--bins N]`:
  convert one or more MHD/RAW volumes to `.npy` (optionally also saving a
  voxel-value histogram PNG).

## 11. Common pitfalls

- **Forgetting `cp.asnumpy`**: `U`/`res` come back as CuPy arrays; every
  downstream consumer (`write_outputs`, napari, numpy comparisons) expects
  numpy. Convert immediately after the solve.
- **Wrong `h`/`Nx_e` for your image shape**: the image must satisfy
  `shape[i] - 1 == N_i * h` exactly along each axis — use `mesh_dims` rather
  than computing this by hand, especially for non-cubic volumes.
  `correlate_gpu` raises `ValueError` if `Nvox`/`Ndof` would overflow
  `int32` (the chunked connectivity arrays are `int32`).
  - **`l0` units**: it's a length in voxels, not a fraction — the convention
  in every example is `l0 = 4.5 * h`, i.e. roughly half an element width at
  the chosen mesh resolution, kept constant in physical units by
  `multiscale_correlate_gpu`'s per-level rescaling. `voxeldvc run`'s
  `--l0-factor` flag (default 4.5) makes this ratio tunable; see §9's
  sensitivity sweep for how accuracy varies with `h`/`l0_factor` and why the
  default is a reasonable choice.
- **`freeze_mask_after` is not a free win**: it fixes boundary 2-cycle
  oscillations on small-strain problems but actively prevents convergence to
  the right answer on large deformations where the active voxel set
  genuinely drifts (see `CLAUDE.md`'s "PA6GF30 CT results" section). Default to leaving it
  off (`dynamic_mask=True`, no freeze) unless you've confirmed your problem
  is the 2-cycle case.
- **`unsafe_voxel_mask` vs `overlap_lost_element_mask`**: don't use the
  former to gate strain output on a porous material — it will flag almost
  everything. Use `overlap_lost_element_mask` (§7).

## 12. Repository layout

- `src/voxeldvc/engine/geometry_dvc.py` — `VoxelCenterStencil`,
  `build_N_stencil`, `build_K_ref` (elastic stiffness), `build_K_ref_laplacian`,
  `build_node_index`, `build_interior_dof_mask`, implicit connectivity helpers.
- `src/voxeldvc/engine/kernels_dvc.py` — matrix-free matvecs (`matvec_H`,
  `matvec_K_ref`, `matvec_equilibrium_gap`), diagonal/Jacobi preconditioners,
  `pcg`, implicit-connectivity index helpers (`get_elem_dofs`,
  `get_elem_voxel_origins`, ...), and the unsafe/overlap-lost mask builders
  (§7).
- `src/voxeldvc/engine/correlate_gpu.py` — `correlate_gpu` (single-level
  GN+PCG loop), `multiscale_correlate_gpu` (pyramid driver, §5.5),
  `fft_rigid_shift`, `affine_prealign`, image gradient/interpolation helpers
  (`image_grad_and_values`, `compute_b_gpu`, `interp_field_vox`).
- `src/voxeldvc/engine/write_output.py` — `write_outputs` (post-solve field
  computation + `.npy`/log writing), `mesh_dims`, `compute_zncc`,
  `compute_principal_strains_cell`/`compute_equivalent_strain`, the per-cell
  quality/material fields, and the deformed-config remaps.
- `src/voxeldvc/vtk_export.py` — `save_dvc_fields_vti`, the guarded
  ParaView/VisIt `.vti` writer (§6) `write_outputs` calls.
- `src/voxeldvc/dvc_defaults.py` — shared defaults (`DEFAULT_H=8`,
  `DEFAULT_L0_FACTOR=4.5`, and the sweep defaults).
- `src/voxeldvc/preprocess.py` (`voxeldvc preprocess`) — affine
  pre-alignment + overlap crop; writes `ref_preprocessed.npy`,
  `def_preprocessed.npy`, `def_ref_aligned.npy`, and `affine_prealign.json`
  into a work directory.
- `src/voxeldvc/run_dvc.py` (`voxeldvc run <work_dir>`) — runs the full
  multiscale DVC pipeline on a preprocessed work directory, producing all
  output fields (§6).
- `src/voxeldvc/run_ground_truth_pipeline.py` (`voxeldvc ground-truth`) —
  the 4-stage ground-truth validation pipeline (§9).
- `src/voxeldvc/ground_truth/` (`voxeldvc sweep-ground-truth`,
  `prepare-ground-truth`, `analyse-ground-truth`,
  `view-displacements-gt-compare`) — the ground-truth generation, sweep,
  comparison, and viewer stages of §9.
- `src/voxeldvc/compute_residuals.py` (`voxeldvc compute-residuals`),
  `crop_numpy_array.py` (`voxeldvc crop-array`), `mhd_to_npy.py`
  (`voxeldvc mhd-to-npy`) — standalone post-/pre-processing utilities (§10).
- `src/voxeldvc/view_*.py` (`voxeldvc view-*`) — napari viewers (§8).
- `tests/test_dvc_gpu_multiscale.py` — multiscale/single-scale/FFT-prealign
  correctness tests.
