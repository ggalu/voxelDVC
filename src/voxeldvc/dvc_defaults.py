# -*- coding: utf-8 -*-
"""Shared CLI defaults for the DVC pipeline scripts, kept in one place so
preprocess.py, run_dvc.py, and run_ground_truth_pipeline.py can't
drift out of sync.
"""

DEFAULT_H = 8
DEFAULT_L0_FACTOR = 4.5

# voxeldvc sweep-ground-truth defaults
DEFAULT_H_SWEEP = "4,8,16"
DEFAULT_L0_FACTOR_SWEEP = "2.25,4.5,9"
