# Integration stencil: trapezoid vs. midpoint quadrature

## Summary

voxelDVC integrates the DVC data-fitting functional element-by-element over
the image voxels using a **vertex-centered trapezoidal** quadrature rule
(nodes coincide with voxels; the default). We implemented and evaluated a
**cell-centered midpoint** rule as an alternative and measured both against
the bundled elastic ground-truth dataset (known displacement field). The
finding, for the record:

> **The trapezoidal rule is more accurate than the midpoint rule on the
> ground-truth benchmark.** The midpoint rule matches the trapezoidal rule in
> the mesh *interior* (as expected from its uniform quadrature weighting), but
> is decisively worse on the *boundary* node layer, which dominates the
> whole-mesh error. On the headline recoverable-node RMS the midpoint rule is
> **~43 % worse** (0.0250 vs. 0.0175 voxels). This boundary deficit is
> *intrinsic* to cell-centered meshing and is **not** removed by cropping the
> mesh away from the image edge.

## The two rules

Both rules integrate each Hex8 element of edge length `h` voxels over image
voxel centres, so the quadrature points coincide with the image grid and no
image interpolation is needed to sample the reference.

- **Trapezoidal (vertex-centered, closed / Newton–Cotes).** Nodes sit *on*
  voxels. Each element integrates over its `(h+1)³` voxels, including the
  boundary layers it shares with neighbours; shared voxels carry trapezoidal
  weights (½, ¼, ⅛ on faces, edges, corners) so every *interior* voxel of the
  global grid sums to unit weight. A mesh of `N` elements per axis spans
  `N·h+1` voxels; node `i` sits at voxel coordinate `i·h`.
- **Midpoint (cell-centered, open).** Nodes are shifted half a voxel so each
  element owns a *unique* block of `h³` interior voxels — a clean partition,
  every voxel counted exactly once with weight 1, no boundary down-weighting.
  A mesh of `N` elements per axis spans `N·h` voxels; node `i` sits at voxel
  coordinate `i·h − ½`, so the two outermost nodes on each axis lie **half a
  voxel outside the image**.

The midpoint rule is theoretically attractive: uniform per-voxel weighting is
the faithful Riemann sum of the matching functional, and unlike the
trapezoidal rule it does not down-weight the outer image boundary. The
experiment below tests whether that translates into better recovered
displacements.

## Benchmark

`voxeldvc ground-truth` runs the standard 4-stage pipeline (prepare →
preprocess → correlate → analyse; see `docs/TUTORIAL.md`) on the bundled
elastic dataset, whose displacement field is known, and scores accuracy by
RMS error of the recovered nodal displacements against ground truth. Both
rules were run at element size `h = 8` with Laplacian regularization and a
dynamic mask; the rule is selected with `--quad-rule {trapezoid,midpoint}`.
The regularization length `l0` was re-optimized independently for each rule
(the midpoint rule's uniform weights change the `H0/L0` calibration), scanning
`l0 = l0_factor · h`.

The trapezoidal baseline reproduces the documented result: recoverable-node
RMS **0.01752 voxels** at `l0_factor = 4.5`.

## Result 1 — whole-mesh (headline) accuracy: trapezoid wins by ~43 %

Recoverable-node RMS error (voxels), best `l0` per rule:

| Rule       | best `l0_factor` | recoverable-node RMS | vs. trapezoid |
|------------|:----------------:|:--------------------:|:-------------:|
| trapezoid  | 4.5              | **0.01752**          | —             |
| midpoint   | 5.0              | 0.02497              | **+42.5 %**   |
| midpoint   | 4.0              | 0.02616              | +49.3 %       |
| midpoint   | 6.0              | 0.02523              | +44.0 %       |

The midpoint rule traces the same U-shaped `l0` accuracy curve with a minimum
near `l0_factor ≈ 5`, but its optimum (0.0250 vox) is ~43 % above the
trapezoidal optimum (0.0175 vox).

## Result 2 — the deficit is entirely in the boundary node layer

Splitting the recoverable nodes into the outer one-node shell (nodes on any
outer mesh face) and the interior (all others):

| Rule (best `l0`)   | interior-node RMS | boundary-shell RMS |
|--------------------|:-----------------:|:------------------:|
| trapezoid (4.5)    | 0.00790           | 0.02488            |
| midpoint (4.0)     | **0.00777**       | 0.03973            |
| midpoint (5.0)     | 0.00880           | 0.03749            |

At its interior-optimal `l0_factor = 4.0` the midpoint rule's **interior is
marginally better** than the trapezoidal rule (0.00777 vs. 0.00790 vox, −1.6 %)
— the predicted benefit of uniform quadrature weighting. But its **boundary
shell is ~50–60 % worse** (0.0397 vs. 0.0249 vox), and because the boundary
shell carries most of the whole-mesh RMS, the midpoint rule loses overall.

This is corroborated by an apples-to-apples per-voxel comparison on the
**identical** set of voxels in the region both meshes cover (the two rules
crop to different sizes, so the recovered fields are resampled onto the same
voxel grid and compared over a shrinking interior margin):

| Common region (identical voxels) | trapezoid | midpoint (4.0) | ratio |
|----------------------------------|:---------:|:--------------:|:-----:|
| full common box (incl. boundary) | 0.01341   | 0.01587        | 1.18  |
| interior (≥4-voxel margin)       | 0.00917   | 0.00936        | 1.02  |
| deep interior (≥8-voxel margin)  | 0.00688   | **0.00682**    | 0.99  |

Deep in the interior the midpoint rule is within 1 % of — and slightly better
than — the trapezoidal rule; the two are statistically equivalent for bulk
measurement. The gap opens only as the boundary is approached.

## Result 3 — the boundary deficit is intrinsic, not fixable by cropping

Because the midpoint mesh reaches one element-layer further into the (ragged,
low-overlap) image boundary than the trapezoidal mesh, we tested whether
insetting the mesh — reserving a band of valid voxels on each side
(`--boundary-margin`) so the mesh stops short of the image edge — recovers the
accuracy. It does not:

| midpoint inset margin | mesh   | headline RMS | interior | boundary |
|-----------------------|:------:|:------------:|:--------:|:--------:|
| 0 voxels              | 8³ el. | 0.02533      | 0.00820  | 0.03828  |
| 4 voxels              | 7³ el. | 0.02513      | 0.01045  | 0.03600  |
| 8 voxels              | 6³ el. | 0.02568      | 0.01051  | 0.03279  |

The headline RMS is flat at ~0.0251 across all insets, and the boundary shell
never approaches the trapezoidal 0.0249. Worse, insetting is
counter-productive: a larger margin coarsens the mesh, so the interior
degrades (0.0082 → 0.0105) about as fast as the boundary improves.

**Interpretation.** The deficit is not caused by the midpoint mesh over-
reaching into poor data (a data-quality effect that cropping would fix). It is
caused by the geometry of the rule itself: a cell-centered mesh *always*
places its outermost nodes half a voxel **beyond** the outermost voxel centre,
regardless of where it is cropped. Those boundary nodes are not quadrature
points and are therefore *extrapolated* from interior data rather than pinned
to a voxel that directly constrains them. A trapezoidal boundary node, by
contrast, sits *on* the outer voxel — it is a quadrature point and is directly
constrained by real image data. Half-voxel extrapolation of an
under-constrained boundary node is the irreducible source of the extra error,
and no cropping strategy can remove it.

## Conclusion

For the recovered displacement field — the quantity the accuracy metric
scores — the vertex-centered **trapezoidal rule is the better choice** and
remains the voxelDVC default. Its slight down-weighting of boundary voxels in
the volume integral is far outweighed by the benefit of anchoring boundary
nodes to real data. The cell-centered **midpoint rule** delivers on its
theoretical promise *only in the interior*, where uniform quadrature weighting
makes it statistically equivalent to (marginally better than) the trapezoidal
rule; but it is ~43 % worse on the boundary-inclusive whole-mesh metric, and
this boundary penalty is intrinsic to cell-centered meshing and cannot be
cropped away.

The midpoint rule remains available (`--quad-rule midpoint`) for uses that
care only about interior/bulk accuracy or require a strict single-weight
per-voxel partition, but it is not recommended for accurate near-boundary
displacement or strain measurement. Achieving whole-mesh parity would require
hybrid boundary elements that re-pin the outer node layer to boundary voxels —
i.e. re-adopting the trapezoidal treatment exactly where it matters.

### Reproduce

```bash
# trapezoid baseline
voxeldvc ground-truth --h 8 --l0-factor 4.5 --quad-rule trapezoid \
    --output-root out_trapezoid
# midpoint (optimum) and an inset attempt
voxeldvc ground-truth --h 8 --l0-factor 5.0 --quad-rule midpoint \
    --output-root out_midpoint
voxeldvc ground-truth --h 8 --l0-factor 4.5 --quad-rule midpoint \
    --boundary-margin 4 --output-root out_midpoint_inset
```

Recoverable-node RMS is reported in each run's `ground_truth_comparison_log.txt`
("Displacement error by region"). All numbers above are `h = 8`, elastic
ground-truth dataset.
