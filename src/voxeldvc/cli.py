# -*- coding: utf-8 -*-
"""voxeldvc: single console-script entry point dispatching to the pipeline
scripts' own main(argv). Each subcommand's module is imported lazily (only
after the subcommand name is known) so e.g. `voxeldvc view-deformed` never
imports cupy and `voxeldvc run` never imports napari.
"""

import argparse
import importlib
import sys

# name -> (module, one-line description shown in `voxeldvc --help`)
_SUBCOMMANDS = {
    "preprocess": ("voxeldvc.preprocess", "affine pre-alignment + overlap crop"),
    "run": ("voxeldvc.run_dvc", "multiscale Gauss-Newton DVC solve"),
    "ground-truth": ("voxeldvc.run_ground_truth_pipeline", "4-stage known-ground-truth accuracy check"),
    "sweep-ground-truth": ("voxeldvc.ground_truth.sweep_ground_truth", "sweep h/l0_factor and score RMS vs. ground truth"),
    "prepare-ground-truth": ("voxeldvc.ground_truth.prepare_ground_truth", "generate ground-truth reference/deformed image pair"),
    "analyse-ground-truth": ("voxeldvc.ground_truth.analyse_ground_truth", "compare recovered vs. ground-truth displacement"),
    "compute-residuals": ("voxeldvc.compute_residuals", "compute per-voxel correlation residual"),
    "crop-array": ("voxeldvc.crop_numpy_array", "crop a saved .npy volume"),
    "mhd-to-npy": ("voxeldvc.mhd_to_npy", "convert an MHD/RAW volume to .npy"),
    "view-deformed": ("voxeldvc.view_deformed", "visualize strains in the deformed configuration"),
    "view-displacements": ("voxeldvc.view_displacements", "visualize the recovered displacement field"),
    "view-displacements-gt-compare": ("voxeldvc.ground_truth.view_displacements_gt_compare", "visualize recovered vs. ground-truth displacement"),
    "view-reference": ("voxeldvc.view_reference", "visualize strains in the reference configuration"),
    "view-overlay": ("voxeldvc.view_overlay_initial", "overlay reference/deformed volumes before DVC"),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    name_width = max(len(name) for name in _SUBCOMMANDS)
    commands_help = "\n".join(
        f"  {name:<{name_width}}  {description}"
        for name, (_, description) in sorted(_SUBCOMMANDS.items())
    )

    parser = argparse.ArgumentParser(
        prog="voxeldvc",
        description="voxelDVC: matrix-free, GPU-accelerated Digital Volume Correlation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"commands:\n{commands_help}",
    )
    parser.add_argument("command", choices=sorted(_SUBCOMMANDS), metavar="command")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    if not argv:
        parser.print_help()
        return 0

    ns = parser.parse_args(argv)

    module_name, _ = _SUBCOMMANDS[ns.command]
    module = importlib.import_module(module_name)
    return module.main(ns.args)


if __name__ == "__main__":
    main()
