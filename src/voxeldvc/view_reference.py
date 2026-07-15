# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-03-18 21:32:20
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-28 17:03:24
import json
import os

import napari
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def main(argv=None):
    # element length used in DVC, read from the preprocessing metadata
    json_path = os.path.join(os.getcwd(), 'affine_prealign.json')
    with open(json_path) as fh:
        meta = json.load(fh)
    lc = meta['h']
    scale = [1/lc] * 3
    # gray-level threshold used by the DVC solve (old JSONs without the key
    # fall back to 0.0, the historical "nonzero intensity is material"
    # convention); voxels at or below glt carry no material / no contrast.
    glt = float(meta.get('glt', 0.0))

    # Von Mises equivalent strain (single deviatoric-magnitude scalar per
    # element) shown by default, with the three principal strains available as
    # additional, initially-hidden channels.
    eq = np.load('equivalent_strain_cell.npy')
    data = np.load('principal_strains_cell.npy')
    print("equivalent_strain shape", eq.shape, "principal_strains shape", data.shape)

    viewer, eq_layer = napari.imshow(eq, name="eps_eq", colormap="turbo",
                                     translate=(0.5, 0.5, 0.5))
    eq_layer.colorbar.visible = True
    eq_layer.interpolation2d = "linear"

    # additional principal-strain channels (hidden by default)
    principal_layers = viewer.add_image(data, channel_axis=3,
                           name=["eps1", "eps2", "eps3"],
                           colormap=["turbo", "turbo", "turbo"],
                           translate=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)])

    for layer in principal_layers:
        layer.colorbar.visible = True
        layer.visible = False
        layer.interpolation2d = "linear"

    # now load the image intensity data file
    im0 = np.load('ref.npy') # yes, this is really the reference image
    #im0 = np.load('../assets/PA6GF30_0.npy')
    print("***SHAPE im0:", im0.shape)
    viewer.add_image(im0, scale=scale, opacity=0.5)

    # now load the unsafe voxel mask from file
    unsafe = ~np.load('unsafe_voxel_mask.npy')
    print("***SHAPE of unsafe mask:", unsafe.shape)
    viewer.add_image(unsafe, scale=scale, opacity=5)

    # gray-level-threshold (glt) material mask, computed the same way as the
    # unsafe overlay above: a per-voxel bool on the reference image where
    # material is present (im0 > glt, the same convention the DVC solve uses to
    # exclude no-material / no-contrast voxels). Shown as True on the material
    # region, matching the inverted unsafe mask (True = trustworthy).
    material_mask = im0 > glt
    print(f"glt = {glt} -> material voxel fraction {material_mask.mean():.3f}")
    viewer.add_image(material_mask, scale=scale, opacity=0.5)

    #viewer.dims.order = (0, 1, 2) # XY plane normal to loading direction
    viewer.dims.order = (1, 2, 0)
    #viewer.dims.order = (2, 0, 1)

    napari.run()


if __name__ == '__main__':
    main()