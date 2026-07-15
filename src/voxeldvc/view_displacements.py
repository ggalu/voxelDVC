# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-03-18 21:32:20
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-16 15:41:55
import napari
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def main(argv=None):
    lc = 8 # element length used in DVC
    scale = [1/lc] * 3

    data = np.load('U_recovered.npy')
    #data = np.load('principal_strains.npy')
    #data = np.load('principal_strains_cell.npy')
    print("shape of displacement data", data.shape)

    #
    # load multichannel image in one line
    viewer, image_layers = napari.imshow(data, channel_axis=3,
                           name=["ux", "uy", "uz"],
                           colormap=["turbo", "turbo", "turbo"])
    #,
    #                       translate=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)])

    for layer in image_layers:
        layer.colorbar.visible = True
        layer.visible = False
        layer.interpolation2d = "linear"
    image_layers[0].visible = True

    # now load the image intensity data file
    im0 = np.load('ref.npy') # yes, this is really the reference image
    #im0 = np.load('../assets/PA6GF30_0.npy')
    print("***SHAPE im0:", im0.shape)
    viewer.add_image(im0, scale=scale, opacity=0.5)

    # now load the residuals
    residuals = np.load('residual.npy')
    print("***SHAPE residuals:", residuals.shape)
    viewer.add_image(residuals, scale=scale, opacity=0.5)

    # now load the unsafe voxel mask from file
    unsafe = ~np.load('unsafe_voxel_mask.npy')
    print("***SHAPE of unsafe mask:", unsafe.shape)
    viewer.add_image(unsafe, scale=scale, opacity=5)

    #viewer.dims.order = (0, 1, 2) # XY plane normal to loading direction
    viewer.dims.order = (1, 2, 0)
    #viewer.dims.order = (2, 0, 1)

    napari.run()


if __name__ == '__main__':
    main()