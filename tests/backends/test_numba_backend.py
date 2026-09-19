import pytest
import numpy as np
from vectorwaves.backends.numba_backend import NumbaMethods, has_numba

def test_numba_backend_presence():
    """Ensures that has_numba matches the import status of Numba on the machine."""
    try:
        import numba
        assert has_numba is True
    except ImportError:
        assert has_numba is False


def test_numba_compute_point_correctness(simple_test_beam):
    backend = NumbaMethods(simple_test_beam)
    
    x, y, z, t = 0.1, 0.2, 0.3, 0.4
    E, D, B = backend.compute_point(x, y, z, t, need_b=True, need_derivs=True)
    
    phase0 = 2.0 * np.pi * z - 2.0 * np.pi * t
    wf0 = np.exp(1j * phase0)
    E0 = np.array([1.0, 0.0, 0.0]) * wf0
    B0 = np.array([0.0, wf0, 0.0])
    
    phase1 = np.pi * y - np.pi * t
    wf1 = np.exp(1j * phase1)
    E1 = np.array([0.0, 0.0, 2.0]) * wf1
    B1 = np.array([2.0 * wf1, 0.0, 0.0])
    
    expected_E = E0 + E1
    expected_B = B0 + B1
    
    assert np.allclose(E, expected_E)
    assert np.allclose(B, expected_B)
    
    dx, dy, dz = D
    assert np.allclose(dx, [0.0, 0.0, 0.0])
    expected_dy = np.array([0.0, 0.0, 2j * np.pi * wf1])
    assert np.allclose(dy, expected_dy)
    expected_dz = np.array([2j * np.pi * wf0, 0.0, 0.0])
    assert np.allclose(dz, expected_dz)


def test_numba_compute_point_options(simple_test_beam):
    backend = NumbaMethods(simple_test_beam)
    x, y, z, t = 0.1, 0.2, 0.3, 0.4
    
    # 1. No B, no derivatives (triggers not need_b and not need_derivs branch)
    E, D, B = backend.compute_point(x, y, z, t, need_b=False, need_derivs=False)
    assert B is None
    assert D == (None, None, None)
    
    # 2. Yes B, no derivatives (triggers need_b and not need_derivs branch)
    E, D, B = backend.compute_point(x, y, z, t, need_b=True, need_derivs=False)
    assert B.shape == (3,)
    assert D == (None, None, None)

    # 3. No B, yes derivatives (triggers not need_b and need_derivs branch)
    E, D, B = backend.compute_point(x, y, z, t, need_b=False, need_derivs=True)
    assert B is None
    assert len(D) == 3
    assert D[0].shape == (3,)


def test_numba_compute_cloud(simple_test_beam):
    backend = NumbaMethods(simple_test_beam)
    
    x = np.array([0.0, 0.1, 0.2])
    y = np.array([0.0, 0.2, 0.4])
    z = np.array([0.0, 0.3, 0.6])
    t = 0.1
    
    E, D, B = backend.compute_cloud(x, y, z, t, need_b=True, need_derivs=True)
    assert E.shape == (3, 3)
    assert B.shape == (3, 3)
    assert len(D) == 3
    for deriv in D:
        assert deriv.shape == (3, 3)
        
    for i in range(3):
        E_pt, D_pt, B_pt = backend.compute_point(x[i], y[i], z[i], t, need_b=True, need_derivs=True)
        assert np.allclose(E[:, i], E_pt)
        assert np.allclose(B[:, i], B_pt)
        assert np.allclose(D[0][:, i], D_pt[0])
        assert np.allclose(D[1][:, i], D_pt[1])
        assert np.allclose(D[2][:, i], D_pt[2])


def test_numba_compute_cloud_options(simple_test_beam):
    backend = NumbaMethods(simple_test_beam)
    x = np.array([0.1, 0.2])
    y = np.array([0.2, 0.3])
    z = np.array([0.3, 0.4])
    t = 0.5
    
    # 1. No B, no derivatives (triggers not need_b and not need_derivs branch)
    E, D, B = backend.compute_cloud(x, y, z, t, need_b=False, need_derivs=False)
    assert B is None
    assert D == (None, None, None)
    
    # 2. Yes B, no derivatives (triggers need_b and not need_derivs branch)
    E, D, B = backend.compute_cloud(x, y, z, t, need_b=True, need_derivs=False)
    assert B.shape == (3, 2)
    assert D == (None, None, None)
    
    # 3. No B, yes derivatives (triggers not need_b and need_derivs branch)
    E, D, B = backend.compute_cloud(x, y, z, t, need_b=False, need_derivs=True)
    assert B is None
    assert len(D) == 3
    assert D[0].shape == (3, 2)


def test_numba_compute_cloud_batching_and_callback(simple_test_beam):
    backend = NumbaMethods(simple_test_beam, max_points_per_batch=2)
    
    x = np.linspace(0, 1, 15)
    y = np.linspace(0, 1, 15)
    z = np.linspace(0, 1, 15)
    t = 0.2
    
    calls = []
    def callback(n):
        calls.append(n)
        
    E, D, B = backend.compute_cloud(x, y, z, t, progress_callback=callback)
    assert calls == [2, 2, 2, 2, 2, 2, 2, 1]
    assert E.shape == (3, 15)


def test_numba_compute_grid(simple_test_beam):
    backend = NumbaMethods(simple_test_beam)
    
    x_vec = np.array([0.0, 0.5])
    y_vec = np.array([0.0, 0.3, 0.6])
    z = 0.2
    t = 0.1
    
    E, D, B = backend.compute_grid(x_vec, y_vec, z, t, need_b=True, need_derivs=True)
    assert E.shape == (3, 3, 2)
    assert B.shape == (3, 3, 2)
    
    for iy, y_val in enumerate(y_vec):
        for ix, x_val in enumerate(x_vec):
            E_pt, D_pt, B_pt = backend.compute_point(x_val, y_val, z, t, need_b=True, need_derivs=True)
            assert np.allclose(E[:, iy, ix], E_pt)
            assert np.allclose(B[:, iy, ix], B_pt)
            assert np.allclose(D[0][:, iy, ix], D_pt[0])
            assert np.allclose(D[1][:, iy, ix], D_pt[1])
            assert np.allclose(D[2][:, iy, ix], D_pt[2])


def test_numba_compute_grid_options(simple_test_beam):
    backend = NumbaMethods(simple_test_beam)
    x_vec = np.array([0.1, 0.2])
    y_vec = np.array([0.2, 0.3])
    z = 0.4
    t = 0.5
    
    # 1. No B, no derivatives (triggers not need_b and not need_derivs branch)
    E, D, B = backend.compute_grid(x_vec, y_vec, z, t, need_b=False, need_derivs=False)
    assert B is None
    assert D == (None, None, None)
    
    # 2. Yes B, no derivatives (triggers need_b and not need_derivs branch)
    E, D, B = backend.compute_grid(x_vec, y_vec, z, t, need_b=True, need_derivs=False)
    assert B.shape == (3, 2, 2)
    assert D == (None, None, None)
    
    # 3. No B, yes derivatives (triggers not need_b and need_derivs branch)
    E, D, B = backend.compute_grid(x_vec, y_vec, z, t, need_b=False, need_derivs=True)
    assert B is None
    assert len(D) == 3
    assert D[0].shape == (3, 2, 2)


def test_numba_compute_grid_batching_and_callback(simple_test_beam):
    backend = NumbaMethods(simple_test_beam, max_points_per_batch=4)
    
    x_vec = np.array([0.0, 0.5])
    y_vec = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    z = 0.0
    t = 0.0
    
    calls = []
    def callback(n):
        calls.append(n)
        
    E, D, B = backend.compute_grid(x_vec, y_vec, z, t, progress_callback=callback)
    assert calls == [2, 2, 1]
    assert E.shape == (3, 5, 2)


def test_numba_maxwell_relations(simple_test_beam):
    backend = NumbaMethods(simple_test_beam)
    
    x, y, z, t = 0.5, -0.2, 1.2, 0.4
    E, D, B = backend.compute_point(x, y, z, t, need_b=True, need_derivs=True)
    dx, dy, dz = D
    
    # Divergence
    div_E = dx[0] + dy[1] + dz[2]
    assert np.isclose(div_E, 0.0, atol=1e-13)
    
    # Curl
    curl_E = np.array([
        dy[2] - dz[1],
        dz[0] - dx[2],
        dx[1] - dy[0]
    ])
    
    k_cross_c = np.cross(simple_test_beam.k.T, simple_test_beam.c.T).T
    phase = np.dot(simple_test_beam.k.T, [x, y, z]) - simple_test_beam.w * t
    wf = np.exp(1j * phase)
    expected_curl = 1j * np.sum(k_cross_c * wf, axis=1)
    
    assert np.allclose(curl_E, expected_curl)