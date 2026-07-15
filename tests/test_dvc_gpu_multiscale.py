# -*- coding: utf-8 -*-
"""Multiscale (coarse-to-fine warm start) test for the matrix-free GPU DVC
port (plan Section 8).

multiscale_correlate_gpu(scales=(...)) decimates f_pix/g_pix by
2**iscale (h_i = h // 2**iscale) and carries the displacement solution
forward as the next finer level's U0, with Ndof fixed across levels (only
h changes). Validates:

  - scales=(0,) (single, full-resolution level) reproduces a plain
    correlate_gpu call exactly.
  - A coarse-to-fine pyramid (scales=(2, 1, 0)) converges to essentially
    the same displacement field as a direct full-resolution run, on a
    small synthetic problem.
  - h not divisible by 2**iscale, or g_origin not divisible by 2**iscale,
    raises ValueError.
"""
import numpy as np

from voxeldvc.engine.geometry_dvc import build_K_ref_laplacian
from voxeldvc.engine.correlate_gpu import correlate_gpu, multiscale_correlate_gpu, fft_rigid_shift


def _build_problem(seed=0):
    """Smooth synthetic texture (low-frequency sinusoids, so coarse-scale
    decimation by up to 4x retains correlatable structure) with a rigid
    1-voxel shift."""
    from scipy.ndimage import shift

    Nx_e = Ny_e = Nz_e = 2
    h = 8
    Nvx = Nx_e * h + 1  # 17

    x = np.arange(Nvx)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    f_pix = (np.sin(2 * np.pi * X / Nvx) + np.sin(2 * np.pi * Y / Nvx)
             + np.sin(2 * np.pi * Z / Nvx) + 0.5 * np.cos(2 * np.pi * (X + Y) / Nvx))
    g_pix = shift(f_pix, (1.0, 0, 0), order=3, mode='nearest')

    return f_pix, g_pix, Nx_e, Ny_e, Nz_e, h


def test_multiscale_single_scale_matches_direct():
    f_pix, g_pix, Nx_e, Ny_e, Nz_e, h = _build_problem()
    K_ref_lap = build_K_ref_laplacian()

    kwargs = dict(K_ref_laplacian=K_ref_lap, l0=4.5 * h, maxiter=20, eps=1e-3, disp=False)

    U_direct, res_direct, _ = correlate_gpu(f_pix, g_pix, Nx_e, Ny_e, Nz_e, h, xp=np, **kwargs)
    # fft_prealign=False: isolate the warm-start mechanism (a single level
    # with scales=(0,) and no prealign must reproduce correlate_gpu exactly).
    U_multi, res_multi, _ = multiscale_correlate_gpu(
        f_pix, g_pix, Nx_e, Ny_e, Nz_e, h, xp=np, scales=(0,),
        fft_prealign=False, **kwargs)

    assert np.allclose(U_direct, U_multi, atol=1e-12)
    assert np.allclose(res_direct, res_multi, atol=1e-12)


def test_multiscale_pyramid_matches_direct():
    f_pix, g_pix, Nx_e, Ny_e, Nz_e, h = _build_problem()
    K_ref_lap = build_K_ref_laplacian()

    kwargs = dict(K_ref_laplacian=K_ref_lap, l0=4.5 * h, maxiter=30, eps=1e-3, disp=False)

    U_direct, _, _ = correlate_gpu(f_pix, g_pix, Nx_e, Ny_e, Nz_e, h, xp=np, **kwargs)
    U_multi, res_multi, _ = multiscale_correlate_gpu(
        f_pix, g_pix, Nx_e, Ny_e, Nz_e, h, xp=np, scales=(2, 1, 0), **kwargs)

    assert np.all(np.isfinite(U_multi))
    assert np.allclose(U_direct, U_multi, atol=1e-2), \
        f"max abs diff = {np.abs(U_direct - U_multi).max()}"


def test_fft_rigid_shift_recovers_known_shift():
    """fft_rigid_shift(f, g) with g = roll(f, d) (circular shift, so the
    phase-correlation assumption holds exactly) must recover d, in the
    convention g(x) = f(x - d)."""
    rng = np.random.default_rng(42)
    f_pix = rng.uniform(0, 1, size=(32, 32, 32))
    d_true = (3, -2, 1)
    g_pix = np.roll(f_pix, d_true, axis=(0, 1, 2))

    d = fft_rigid_shift(f_pix, g_pix, np)
    assert np.allclose(d, d_true)


def test_multiscale_rejects_indivisible_h():
    f_pix, g_pix, Nx_e, Ny_e, Nz_e, h = _build_problem()  # h=8
    K_ref_lap = build_K_ref_laplacian()

    try:
        multiscale_correlate_gpu(f_pix, g_pix, Nx_e, Ny_e, Nz_e, h, xp=np,
                                  scales=(4, 0), K_ref_laplacian=K_ref_lap,
                                  l0=4.5 * h, disp=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for h=8 not divisible by 2**4")


def test_multiscale_rejects_indivisible_g_origin():
    f_pix, g_pix, Nx_e, Ny_e, Nz_e, h = _build_problem()  # h=8
    K_ref_lap = build_K_ref_laplacian()

    try:
        multiscale_correlate_gpu(f_pix, g_pix, Nx_e, Ny_e, Nz_e, h, xp=np,
                                  scales=(1, 0), g_origin=(1, 0, 0),
                                  K_ref_laplacian=K_ref_lap, l0=4.5 * h, disp=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for g_origin=(1,0,0) not "
                              "divisible by 2**1")


if __name__ == '__main__':
    test_multiscale_single_scale_matches_direct()
    test_multiscale_pyramid_matches_direct()
    test_fft_rigid_shift_recovers_known_shift()
    test_multiscale_rejects_indivisible_h()
    test_multiscale_rejects_indivisible_g_origin()
    print("All multiscale tests passed.")
