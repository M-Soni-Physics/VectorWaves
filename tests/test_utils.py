import pytest
import numpy as np
from vectorwaves.utils import get_stokes_params, get_pol_ellipse_params, decompose_in_basis

# =========================================================================
#                    1. STOKES PARAMETERS TESTS
# =========================================================================

def test_stokes_params_horizontal_linear():
    E1 = np.array([1.0])
    E2 = np.array([0.0])
    
    res = get_stokes_params(E1, E2, normalize=False)
    assert np.isclose(res['S0'], 1.0)
    assert np.isclose(res['S1'], 1.0)
    assert np.isclose(res['S2'], 0.0)
    assert np.isclose(res['S3'], 0.0)

def test_stokes_params_vertical_linear():
    E1 = np.array([0.0])
    E2 = np.array([1.0])
    
    res = get_stokes_params(E1, E2, normalize=False)
    assert np.isclose(res['S0'], 1.0)
    assert np.isclose(res['S1'], -1.0)
    assert np.isclose(res['S2'], 0.0)
    assert np.isclose(res['S3'], 0.0)

def test_stokes_params_plus_45_linear():
    E1 = np.array([1.0 / np.sqrt(2)])
    E2 = np.array([1.0 / np.sqrt(2)])
    
    res = get_stokes_params(E1, E2, normalize=True)
    assert np.isclose(res['S0'], 1.0)
    assert np.isclose(res['s1'], 0.0)
    assert np.isclose(res['s2'], 1.0)
    assert np.isclose(res['s3'], 0.0)

def test_stokes_params_circular():
    E1 = np.array([1.0 / np.sqrt(2)])
    E2 = np.array([1j / np.sqrt(2)]) # Phase difference: +pi/2
    
    res = get_stokes_params(E1, E2, normalize=True)
    assert np.isclose(res['S0'], 1.0)
    assert np.isclose(res['s1'], 0.0)
    assert np.isclose(res['s2'], 0.0)
    assert np.isclose(res['s3'], -1.0) # S3 = 2*Imag(E1 * E2*) = 2*Imag(1/sqrt(2) * -1j/sqrt(2)) = -1.0

def test_stokes_zero_intensity():
    E1 = np.array([0.0])
    E2 = np.array([0.0])
    
    res = get_stokes_params(E1, E2, normalize=True)
    assert res['S0'] == 0.0
    assert res['s1'] == 0.0
    assert res['s2'] == 0.0
    assert res['s3'] == 0.0

# =========================================================================
#                    2. POLARIZATION ELLIPSE TESTS
# =========================================================================

def test_pol_ellipse_linear():
    E1 = np.array([1.0])
    E2 = np.array([1.0])
    
    res = get_pol_ellipse_params(E1, E2)
    assert np.isclose(res['psi'], np.pi / 4) # orientation at 45 degrees
    assert np.isclose(res['chi'], 0.0)       # zero ellipticity
    assert np.isclose(res['delta'], 0.0)     # zero phase difference
    assert np.isclose(res['b'], 0.0)         # semi-minor axis is zero

def test_pol_ellipse_circular():
    E1 = np.array([1.0])
    E2 = np.array([1j])
    
    res = get_pol_ellipse_params(E1, E2)
    assert np.isclose(res['chi'], -np.pi / 4)  # Chi is -45 degrees for pure circular
    assert np.isclose(res['delta'], np.pi / 2) # Delta = phase(E2) - phase(E1) = pi/2
    assert np.isclose(res['a'], res['b'])      # semi-major equals semi-minor
    assert np.isclose(res['handedness'], -1.0) # S3 is negative

# =========================================================================
#                    3. BASIS DECOMPOSITION TESTS
# =========================================================================

def test_decompose_in_basis_identity():
    E1 = np.array([1.0 + 0j, 0.0 + 0j])
    E2 = np.array([0.0 + 0j, 1.0 + 0j])
    
    # Decomposing into the identity basis
    res = decompose_in_basis(E1, E2, u=(1, 0))
    assert np.allclose(res['u_hat'], [1.0, 0.0])
    assert np.allclose(res['v_hat'], [0.0, 1.0])
    assert np.allclose(res['E_u'], E1)
    assert np.allclose(res['E_v'], E2)

def test_decompose_in_basis_rotated():
    E1 = np.array([1.0 + 0j])
    E2 = np.array([1.0 + 0j])
    
    # Rotated by 45 degrees
    u = np.array([1.0, 1.0]) / np.sqrt(2)
    res = decompose_in_basis(E1, E2, u=u)
    
    # E is entirely along u: E_u = sqrt(2) and E_v = 0
    assert np.allclose(res['E_u'], [np.sqrt(2)])
    assert np.allclose(res['E_v'], [0.0])