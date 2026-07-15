# `assets/` — bundled datasets and images

Data shipped with voxelDVC: a real CT image pair used by the quick-start /
tutorial, two synthetic known-displacement datasets used by the ground-truth
validation pipeline, and the project icon. The `.npy` paths here are resolved
relative to the repository root, so run the commands below from there.

## PA6GF30 CT pair — the tutorial dataset

Micro-CT scans of a PA6GF30 (polyamide-6 with 30 % short glass fibre)
specimen: `PA6GF30_0` is the reference (undeformed) state, `PA6GF30_1` the
deformed state.

| File | Shape | dtype | Size | Role |
|------|-------|-------|------|------|
| `PA6GF30_0.npy` | `(193, 193, 193)` | `uint16` | 14 M | reference image `f(x)` |
| `PA6GF30_1.npy` | `(193, 193, 193)` | `uint16` | 14 M | deformed image `g(x)` |

Used by the quick-start in `README.md` and the walkthrough in
`docs/TUTORIAL.md`:

```bash
voxeldvc preprocess assets/PA6GF30_0.npy assets/PA6GF30_1.npy --output-dir test
```

There is no bundled ground-truth displacement for this pair — it is real data,
so the true field is unknown. See `CLAUDE.md`'s "PA6GF30 CT results" section
for known correlation-quality notes on this dataset.

## Ground-truth validation datasets

Synthetic datasets with a *known* nodal displacement field, consumed by the
4-stage ground-truth pipeline (`voxeldvc ground-truth`, `docs/TUTORIAL.md` §9)
to score DVC accuracy by node RMS displacement error. The reference image is
the same `65³` volume in both; only the applied displacement field differs.
The directory names encode the nominal z-strain: `z+0.01` = +1 % tension,
`z-0.05` = −5 % compression.

Each dataset contains:

| File | Shape | dtype | Size | Role |
|------|-------|-------|------|------|
| `ref.npy` | `(65, 65, 65)` | `uint16` | 540 K | reference image (identical across both datasets) |
| `U.npy` | `(66, 66, 66, 3)` | `float32` | 3.3 M | ground-truth nodal displacement `(ux, uy, uz)`, voxel units |

`U.npy` is sampled at integer coordinates `0..N` (`N+1 = 66` points per axis) —
one grid point more per axis than the `N = 65` image voxels — so trilinear
interpolation of the field onto any image voxel never extrapolates (see
`src/voxeldvc/ground_truth/prepare_ground_truth.py`). Displacements are
channels-last `(ux, uy, uz)`.

### `GT_z+0.01/` — tension

`uz` ranges `0 → +0.65` voxels over the 65-voxel height (≈ +1 % z-strain);
`ux, uy` small. This is the **default** ground-truth dataset — the pipeline
falls back to it when `--gt-ref-file` / `--gt-displacement-file` are omitted
(hardcoded in `prepare_ground_truth.py` as `GT_DIR = assets/GT_z+0.01`).

```bash
voxeldvc ground-truth --gt-ref-file assets/GT_z+0.01/ref.npy \
                      --gt-displacement-file assets/GT_z+0.01/U.npy
```

### `GT_z-0.05/` — compression

`uz` ranges `0 → −3.25` voxels (≈ −5 % z-strain) — a larger deformation than
the tension case. Passed explicitly (it is not a default):

```bash
voxeldvc ground-truth --gt-ref-file assets/GT_z-0.05/ref.npy \
                      --gt-displacement-file assets/GT_z-0.05/U.npy
```

| Extra file | Shape | dtype | Size | Role |
|------------|-------|-------|------|------|
| `def.npy` | `(65, 65, 65)` | `uint16` | 540 K | pre-computed deformed image `ref` warped by `U` |

`def.npy` is a bundled copy of the deformed volume for inspection. The
ground-truth pipeline does **not** read it — it regenerates the deformed image
from `ref.npy` + `U.npy` itself (`deformed(x) = ref(x − u_gt(x))`), so `def.npy`
is provided only for convenience / reference. (`GT_z+0.01/` ships no `def.npy`.)

## Project icon

| File | Size | Role |
|------|------|------|
| `voxeldvc_icon_tensile.svg` | 24 K | project logo — reference volume `f(x)` mapped to a tensile-necked deformed volume `g(x+u)`; embedded in `README.md` |
