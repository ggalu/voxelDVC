# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-07-07 18:10:49
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-07-08 10:07:21
"""Generate the vector figures for the voxelDVC / Experimental Mechanics manuscript.

Two figures are produced, according to the information in STENCIL.md

both from information that exists in the repository
documentation (docs/TUTORIAL.md, docs/CLAUDE.md) rather than from any solver
run:

  1. mesh_voxel_schematic.pdf
     This follows the **vertex-centered trapezoidal** quadrature rule.

  2. mesh_cell_schematic.pdf
     This follows the **cell-centered midpoint** rule.

  3. mesh_comparison.pdf
     Both of the above side by side in one figure, titled
     "vertex centered" and "cell centered".

Run:  python figures/make_figures.py
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# A neutral, print-friendly palette.
INK = "#1a1a1a"
NODE = "#c0392b"
VOXEL = "#00a03c"
ELEM_FILL = "#eef3f8"
ELEM_EDGE = "#34495e"
GRID = "#9aa7b1"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    }
)


def _draw_schematic(ax, *, rule):
    """Draw one 2D node/element/voxel schematic onto ``ax``.

    The two quadrature stencils share every colour, marker size and label
    position; only the geometry differs, so both are drawn here and selected
    with ``rule``:

    - ``rule="vertex"`` -- vertex-centered *trapezoidal* rule. Nodes sit *on*
      voxel centres (node ``i`` coincides with voxel ``i*h``), element edges
      run through those centres, an ``ne``-element axis spans ``ne*h + 1``
      voxels, and each element covers ``(h+1)**2`` voxels.
    - ``rule="cell"`` -- cell-centered *midpoint* rule. Nodes sit on the
      voxel-block corners (node ``i`` at voxel coordinate ``i*h - 1/2``, i.e.
      drawing coordinate ``i*h``), so the outer nodes lie half a voxel outside
      the image; element edges run *between* voxels, an ``ne``-element axis
      spans ``ne*h`` voxels, and each element owns a unique block of ``h**2``
      voxels.

    All descriptive labels are placed in the margin to the right of the mesh
    and connected with leader lines so that no text overlaps the voxel grid.
    """
    # ---- tunable layout constants -----------------------------------------
    h2d = 3               # voxels per element edge
    ne = 2                # elements per axis

    if rule == "vertex":
        npix = h2d * ne + 1   # voxels per axis for the closed trapezoidal rule
        node_off = 0.5        # node sits on the voxel centre
        elem_voxels = "$(h+1)^2$"
        node_text_dx = 0.5    # extra x offset for the "node" label text
        elem_xy_dy = 0.0      # y nudge for the "element" label target
    elif rule == "cell":
        npix = h2d * ne       # voxels per axis for the open midpoint rule
        node_off = 0.0        # node sits on the voxel-block corner
        elem_voxels = "$h^2$"
        node_text_dx = 1.0
        elem_xy_dy = 0.3
    else:
        raise ValueError("rule must be 'vertex' or 'cell', got %r" % (rule,))

    margin_r = 3.4         # blank space to the right of the mesh (holds labels)
    margin_top = 0.5       # blank space above the mesh
    margin_bot = 0.5       # blank space below the mesh
    margin_l = 0.5         # blank space to the left of the mesh

    # voxel grid cells
    for i in range(npix):
        for j in range(npix):
            ax.add_patch(
                Rectangle((i, j), 1, 1, facecolor=ELEM_FILL,
                          edgecolor=GRID, lw=0.6)
            )
    # voxel centers = quadrature points
    for i in range(npix):
        for j in range(npix):
            ax.plot(i + 0.5, j + 0.5, "o", color=VOXEL, ms=6.0, zorder=10)

    # element boundaries (thick): through voxel centres (vertex) or between
    # voxels (cell), spanning the outermost nodes on each axis.
    edge_lo = node_off
    edge_hi = ne * h2d + node_off
    for e in range(ne + 1):
        c = e * h2d + node_off
        ax.plot([c, c], [edge_lo, edge_hi], color=ELEM_EDGE, lw=2.6, zorder=5)
        ax.plot([edge_lo, edge_hi], [c, c], color=ELEM_EDGE, lw=2.6, zorder=5)

    # nodes
    for i in range(ne + 1):
        for j in range(ne + 1):
            ax.plot(i * h2d + node_off, j * h2d + node_off, "s", color=NODE,
                    ms=18, markeredgecolor=NODE, markeredgewidth=0.9, zorder=7)

    # ---- labels, all placed in the margin outside the mesh ----------------
    # "node": point at the right node at centre height; text sits right.
    ax.annotate("node", xy=(ne * h2d + node_off, h2d + node_off),
                xytext=(npix + node_text_dx, h2d),
                color=NODE, fontsize=11, fontweight="bold",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=NODE, lw=1.0,
                                shrinkA=2, shrinkB=14))

    # "element": point at the top-right element centre; text sits to the right.
    ax.annotate("element with $h=3$\n%s voxels" % elem_voxels,
                xy=(ne * h2d + node_off - h2d / 2,
                    ne * h2d + node_off - h2d / 2 + elem_xy_dy),
                xytext=(npix + 0.5, npix - h2d / 2),
                color=ELEM_EDGE, fontsize=11,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=ELEM_EDGE, lw=1.0,
                                shrinkA=2, shrinkB=2))

    # "voxel center (quadrature point)": point at a plain (non-node) voxel on
    # the bottom row; text to the right.
    ax.annotate("voxel center\n(quadrature point)", xy=(npix - 1.5, 1.5),
                xytext=(npix + 0.5, 0.5), color=VOXEL, fontsize=11,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=VOXEL, lw=1.0,
                                shrinkA=2, shrinkB=4))

    ax.set_xlim(-margin_l, npix + margin_r)
    ax.set_ylim(-margin_bot, npix + margin_top)
    ax.set_aspect("equal")
    ax.axis("off")


def mesh_vertex_centred_schematic():
    """Vertex-centered trapezoidal schematic, saved as its own PDF."""
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    _draw_schematic(ax, rule="vertex")
    out = os.path.join(HERE, "mesh_voxel_schematic.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", out)


def mesh_cell_centred_schematic():
    """Cell-centered midpoint schematic, saved as its own PDF."""
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    _draw_schematic(ax, rule="cell")
    out = os.path.join(HERE, "mesh_cell_schematic.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", out)


def mesh_comparison_schematic():
    """Both stencils side by side in one PDF, with a title over each panel."""
    fig, (ax_v, ax_c) = plt.subplots(1, 2, figsize=(14.4, 6.8))
    _draw_schematic(ax_v, rule="vertex")
    _draw_schematic(ax_c, rule="cell")
    ax_v.set_title("vertex centered", fontsize=13, fontweight="bold", pad=10)
    ax_c.set_title("cell centered", fontsize=13, fontweight="bold", pad=10)

    out = os.path.join(HERE, "mesh_comparison.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    mesh_vertex_centred_schematic()
    mesh_cell_centred_schematic()
    mesh_comparison_schematic()