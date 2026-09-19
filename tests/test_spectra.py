import pytest
import numpy as np
from vectorwaves.spectra import KSpaceSpectra, PolychromaticSpectra

# =========================================================================
#                       1. K-SPACE SPECTRA TESTS
# =========================================================================

def test_kspace_uniform():
    # Test 1D input (single vector)
    k_1d = np.array([0.0, 0.0, 1.0])
    res_1d = KSpaceSpectra.uniform(k_1d)
    assert res_1d == 1.0 + 0j
    assert isinstance(res_1d, complex)

    # Test 2D input (vectorized)
    k_2d = np.array([[0, 1], [0, 1], [1, 1]])
    res_2d = KSpaceSpectra.uniform(k_2d)
    assert res_2d.shape == (2,)
    assert np.all(res_2d == 1.0 + 0j)


def test_kspace_gaussian():
    # Evaluate at origin (max) and at 1 sigma
    sigma = 2.0
    k_vec = np.array([
        [0.0, sigma],  # kx
        [0.0, 0.0],    # ky
        [1.0, 1.0]     # kz
    ])
    res = KSpaceSpectra.gaussian(k_vec, sigma_k_perp=sigma)
    
    assert np.isclose(res[0], 1.0 + 0j)  # peak at origin
    assert np.isclose(res[1], np.exp(-0.5) + 0j) # at 1 sigma distance
    assert res.dtype == complex


def test_kspace_tophat():
    k_perp_max = 1.5
    k_vec = np.array([
        [0.0, 1.4, 1.6],  # kx
        [0.0, 0.0, 0.0],  # ky
        [1.0, 1.0, 1.0]   # kz
    ])
    res = KSpaceSpectra.tophat(k_vec, k_perp_max=k_perp_max)
    
    assert res[0] == 1.0 + 0j  # Well inside
    assert res[1] == 1.0 + 0j  # Just inside
    assert res[2] == 0.0 + 0j  # Just outside


def test_kspace_laguerre_gauss_branches():
    sigma = 1.0
    
    # 1. Test 1D fallback (`size == 1` branch)
    k_1d = np.array([1.0, 0.0, 1.0])
    res_1d = KSpaceSpectra.laguerre_gauss(k_1d, p=0, l=1, sigma_k_perp=sigma)
    assert isinstance(res_1d, (complex, np.complexfloating))
    
    # 2. Test Origin division-by-zero handling (`np.where(k_perp_sq == 0)` branch)
    k_origin = np.array([
        [0.0], [0.0], [1.0]
    ])
    res_origin = KSpaceSpectra.laguerre_gauss(k_origin, p=0, l=1, sigma_k_perp=sigma)
    assert res_origin[0] == 0j

    # 3. Test Vectorized valid output
    k_vec = np.array([
        [1.0, 0.0], 
        [0.0, 1.0], 
        [1.0, 1.0]
    ])
    res_vec = KSpaceSpectra.laguerre_gauss(k_vec, p=1, l=2, sigma_k_perp=sigma)
    assert res_vec.shape == (2,)


def test_kspace_hermite_gauss():
    sigma = 1.0
    k_vec = np.array([
        [0.0, 1.0], 
        [0.0, 1.0], 
        [1.0, 1.0]
    ])
    res = KSpaceSpectra.hermite_gauss(k_vec, l=1, m=1, sigma_k_perp=sigma)
    assert res.shape == (2,)
    assert res.dtype == complex


def test_kspace_bessel_gauss_branches():
    theta_0 = np.pi / 4
    sigma = 0.1
    
    # K-vectors: [Origin, On-cone (z-axis only), Off-cone]
    k_vec = np.array([
        [0.0, 0.0, 1.0],  # kx
        [0.0, 0.0, 0.0],  # ky
        [0.0, 1.0, 1.0]   # kz
    ])
    
    # Branch 1: l = 0 (no vortex)
    res_l0 = KSpaceSpectra.bessel_gauss(k_vec, theta_0, sigma_theta=sigma, l=0)
    assert res_l0[0] == 0j  # k_mag == 0 branch masked to 0
    assert res_l0.shape == (3,)
    
    # Branch 2: l != 0 (vortex branch)
    res_l1 = KSpaceSpectra.bessel_gauss(k_vec, theta_0, sigma_theta=sigma, l=1)
    assert res_l1[0] == 0j
    assert res_l1.shape == (3,)


# =========================================================================
#                       2. POLYCHROMATIC SPECTRA TESTS
# =========================================================================

def test_poly_uniform():
    assert PolychromaticSpectra.uniform(400.0) == 1.0
    assert PolychromaticSpectra.uniform(800.0) == 1.0


def test_poly_gaussian():
    # Peak should be 1.0
    assert np.isclose(PolychromaticSpectra.gaussian(500.0, center=500.0, sigma=10.0), 1.0)
    # 1 sigma away should be exp(-0.5)
    res_1sig = PolychromaticSpectra.gaussian(510.0, center=500.0, sigma=10.0)
    assert np.isclose(res_1sig, np.exp(-0.5))


def test_poly_lorentzian():
    # Peak should be 1.0
    assert np.isclose(PolychromaticSpectra.lorentzian(500.0, center=500.0, gamma=10.0), 1.0)
    # At HWHM (center + gamma), amplitude drops to 0.5
    assert np.isclose(PolychromaticSpectra.lorentzian(510.0, center=500.0, gamma=10.0), 0.5)


def test_poly_tophat():
    center = 500.0
    width = 20.0
    
    assert PolychromaticSpectra.tophat(500.0, center, width) == 1.0   # Dead center
    assert PolychromaticSpectra.tophat(510.0, center, width) == 1.0   # On edge
    assert PolychromaticSpectra.tophat(511.0, center, width) == 0.0   # Just outside