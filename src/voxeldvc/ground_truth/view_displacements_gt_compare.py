# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-07-05 19:59:34
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-07-12 10:20:25

# view_displacements_gt_compare.py: napari viewer comparing the recovered
# displacement field to the prescribed (ground-truth) displacement field,
# voxel by voxel. Run from the work directory written by
# run_ground_truth_pipeline.py (e.g. dvc_elastic_groundtruth/), which must
# contain U_recovered_vox.npy, U_gt_vox.npy and U_error_vox.npy (all
# (nx,ny,nz,3)) -- written by analyse_ground_truth.py.
import napari
import numpy as np


def main(argv=None):
    lc = 8  # element length used in DVC
    scale = [1/lc] * 3

    recovered = np.load('U_recovered_vox.npy')
    gt = np.load('U_gt_vox.npy')
    error = np.load('U_error_vox.npy')
    print("shape of recovered/gt/error voxel displacement data:",
          recovered.shape, gt.shape, error.shape)

    viewer, recovered_layers = napari.imshow(
        recovered, channel_axis=3,
        name=["ux_recovered", "uy_recovered", "uz_recovered"],
        colormap=["turbo", "turbo", "turbo"])

    _, gt_layers = napari.imshow(
        gt, channel_axis=3, viewer=viewer,
        name=["ux_gt", "uy_gt", "uz_gt"],
        colormap=["turbo", "turbo", "turbo"])

    _, error_layers = napari.imshow(
        error, channel_axis=3, viewer=viewer,
        name=["ux_error", "uy_error", "uz_error"],
        colormap=["turbo", "turbo", "turbo"])

    for layer in recovered_layers + gt_layers + error_layers:
        layer.colorbar.visible = True
        layer.visible = False
        layer.interpolation2d = "linear"
    recovered_layers[0].visible = True
    #gt_layers[0].visible = True
    #error_layers[0].visible = True

    # now load the image intensity data file for spatial context
    im0 = np.load('ref.npy')
    print("***SHAPE im0:", im0.shape)
    #viewer.add_image(im0, scale=scale, opacity=0.5)
    viewer.add_image(im0)

    # voxels whose GT-deformed position falls outside the deformed image --
    # error there reflects missing data in g, not DVC accuracy (see
    # analyse_ground_truth.py's voxel_in_bounds/vox_recoverable)
    #unrecoverable = ~np.load('voxel_recoverable_mask.npy')
    #print("***SHAPE of unrecoverable mask:", unrecoverable.shape)
    #viewer.add_image(unrecoverable, scale=scale, opacity=0.5)

    viewer.dims.order = (1, 2, 0)

    napari.run()


if __name__ == '__main__':
    main()
