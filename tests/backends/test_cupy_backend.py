import pytest
import numpy as np
from vectorwaves.backends.cupy_backend import CupyMethods, has_cupy

def test_cupy_unavailability_error():
    """If CuPy/CUDA is missing, instantiating CupyMethods must trigger a RuntimeError."""
    if not has_cupy:
        class DummyBeam:
            k = np.zeros((3, 1))
            w = np.ones(1)
            inv_w = np.ones(1)
            c = np.zeros((3, 1), dtype=complex)
        
        with pytest.raises(RuntimeError, match="CuPy/CUDA not found"):
            CupyMethods(DummyBeam())
    else:
        pytest.skip("CuPy is present on this machine, skipping the error-branch test.")


@pytest.mark.skipif(not has_cupy, reason="CuPy/CUDA is not installed or available on this system.")
class TestCupyBackend:
    
    @pytest.mark.parametrize("single_precision", [True, False])
    def test_cupy_compute_point_correctness(self, simple_test_beam, single_precision):
        backend = CupyMethods(simple_test_beam, use_single_precision=single_precision)
        
        x, y, z, t = 0.1, 0.2, 0.3, 0.4
        E, D, B = backend.compute_point(x, y, z, t, need_b=True, need_derivs=True)
        
        # Calculate expected analytical values
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
        
        # Adjust analytical tolerance requirements dynamically based on precision
        tol = 1e-6 if single_precision else 1e-12
        
        assert np.allclose(E, expected_E, atol=tol, rtol=tol)
        assert np.allclose(B, expected_B, atol=tol, rtol=tol)
        
        dx, dy, dz = D
        assert np.allclose(dx, [0.0, 0.0, 0.0], atol=tol)
        expected_dy = np.array([0.0, 0.0, 2j * np.pi * wf1])
        assert np.allclose(dy, expected_dy, atol=tol, rtol=tol)
        expected_dz = np.array([2j * np.pi * wf0, 0.0, 0.0])
        assert np.allclose(dz, expected_dz, atol=tol, rtol=tol)

    @pytest.mark.parametrize("single_precision", [True, False])
    def test_cupy_compute_point_options(self, simple_test_beam, single_precision):
        backend = CupyMethods(simple_test_beam, use_single_precision=single_precision)
        x, y, z, t = 0.1, 0.2, 0.3, 0.4
        
        E, D, B = backend.compute_point(x, y, z, t, need_b=False, need_derivs=False)
        assert B is None
        assert D == (None, None, None)
        
        E, D, B = backend.compute_point(x, y, z, t, need_b=True, need_derivs=False)
        assert B.shape == (3,)
        assert D == (None, None, None)

    @pytest.mark.parametrize("single_precision", [True, False])
    def test_cupy_compute_cloud(self, simple_test_beam, single_precision):
        backend = CupyMethods(simple_test_beam, use_single_precision=single_precision)
        
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
            
        tol = 1e-6 if single_precision else 1e-12
        for i in range(3):
            E_pt, D_pt, B_pt = backend.compute_point(x[i], y[i], z[i], t, need_b=True, need_derivs=True)
            assert np.allclose(E[:, i], E_pt, atol=tol)
            assert np.allclose(B[:, i], B_pt, atol=tol)
            assert np.allclose(D[0][:, i], D_pt[0], atol=tol)
            assert np.allclose(D[1][:, i], D_pt[1], atol=tol)
            assert np.allclose(D[2][:, i], D_pt[2], atol=tol)

    @pytest.mark.parametrize("single_precision", [True, False])
    def test_cupy_compute_cloud_progress_callback(self, simple_test_beam, single_precision):
        backend = CupyMethods(simple_test_beam, use_single_precision=single_precision)
        
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        t = 0.2
        
        calls = []
        def callback(n):
            calls.append(n)
            
        E, D, B = backend.compute_cloud(x, y, z, t, progress_callback=callback)
        # CHUNK is 500_000, so 5 points are evaluated in a single step
        assert calls == [5]
        assert E.shape == (3, 5)

    @pytest.mark.parametrize("single_precision", [True, False])
    def test_cupy_compute_grid(self, simple_test_beam, single_precision):
        backend = CupyMethods(simple_test_beam, use_single_precision=single_precision)
        
        x_vec = np.array([0.0, 0.5])
        y_vec = np.array([0.0, 0.3, 0.6])
        z = 0.2
        t = 0.1
        
        E, D, B = backend.compute_grid(x_vec, y_vec, z, t, need_b=True, need_derivs=True)
        assert E.shape == (3, 3, 2)
        assert B.shape == (3, 3, 2)
        
        tol = 1e-6 if single_precision else 1e-12
        for iy, y_val in enumerate(y_vec):
            for ix, x_val in enumerate(x_vec):
                E_pt, D_pt, B_pt = backend.compute_point(x_val, y_val, z, t, need_b=True, need_derivs=True)
                assert np.allclose(E[:, iy, ix], E_pt, atol=tol)
                assert np.allclose(B[:, iy, ix], B_pt, atol=tol)
                assert np.allclose(D[0][:, iy, ix], D_pt[0], atol=tol)
                assert np.allclose(D[1][:, iy, ix], D_pt[1], atol=tol)
                assert np.allclose(D[2][:, iy, ix], D_pt[2], atol=tol)

    @pytest.mark.parametrize("single_precision", [True, False])
    def test_cupy_compute_grid_progress_callback(self, simple_test_beam, single_precision):
        backend = CupyMethods(simple_test_beam, use_single_precision=single_precision)
        
        x_vec = np.array([0.0, 0.5])
        y_vec = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        z = 0.0
        t = 0.0
        
        calls = []
        def callback(n):
            calls.append(n)
            
        E, D, B = backend.compute_grid(x_vec, y_vec, z, t, progress_callback=callback)
        assert sum(calls) == 5
        assert E.shape == (3, 5, 2)

    def test_cupy_maxwell_relations(self, simple_test_beam):
        # Explicitly evaluate double-precision mode for Maxwell verification
        backend = CupyMethods(simple_test_beam, use_single_precision=False)
        x, y, z, t = 0.5, -0.2, 1.2, 0.4
        E, D, B = backend.compute_point(x, y, z, t, need_b=True, need_derivs=True)
        dx, dy, dz = D
        
        # Divergence
        div_E = dx[0] + dy[1] + dz[2]
        assert np.isclose(div_E, 0.0, atol=1e-12)
        
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
        assert np.allclose(curl_E, expected_curl, atol=1e-12)