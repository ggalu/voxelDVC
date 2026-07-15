# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-18 20:11:45
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-18 20:32:54
#!/usr/bin/env python3
"""
view_overlay_MR4.py

Napari overlay of the cropped MR4 reference and deformed CT volumes.

  Reference  →  magenta/pink   (additive, colormap 'magenta')
  Deformed   →  green          (additive, colormap 'green')
  Overlap    →  gray/white     (magenta + green = white in additive RGB)

Contrast limits are set independently per image using the 1st–99th percentile
of non-zero voxels, so both channels appear with similar apparent brightness.

Run from any directory; paths are absolute.
"""
import os
import napari
import numpy as np


def percentile_clim(data, lo=1, hi=99):
    """Robust contrast limits from non-zero voxels."""
    nz = data[data > 0]
    if nz.size == 0:
        return float(data.min()), float(data.max())
    return float(np.percentile(nz, lo)), float(np.percentile(nz, hi))


def main(argv=None):
    ref_path = 'ref_preprocessed.npy'
    def_path = 'def_preprocessed.npy'

    print(f"Loading {ref_path}")
    ref = np.load(ref_path)
    print(f"  ref: shape={ref.shape}  dtype={ref.dtype}")

    print(f"Loading {def_path}")
    deformed = np.load(def_path)
    print(f"  def: shape={deformed.shape}  dtype={deformed.dtype}")

    clim_ref = percentile_clim(ref)
    clim_def = percentile_clim(deformed)
    print(f"  contrast limits  ref={clim_ref}  def={clim_def}")

    viewer = napari.Viewer(title='overlay  –  pink=reference  green=deformed  gray=overlap')

    viewer.add_image(ref,      name='reference (pink/magenta)',
                     colormap='magenta', blending='additive',
                     contrast_limits=clim_ref)
    viewer.add_image(deformed, name='deformed (green)',
                     colormap='green',   blending='additive',
                     contrast_limits=clim_def)

    # default slice orientation: axis-1 vertical, axis-2 horizontal
    viewer.dims.order = (0, 1, 2)

    napari.run()


if __name__ == '__main__':
    main()
