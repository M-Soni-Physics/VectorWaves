import pytest
import numpy as np
import warnings

from vectorwaves.config_stuff import get_config
from vectorwaves.beam_stuff import Beam, BeamMaker

# =========================================================================
#                       FIXTURES
# =========================================================================

@pytest.fixture
def base_config():
    """Provides a deterministic configuration for testing."""
    cfg = get_config()
    cfg.source.randomize.off()
    cfg.source.num_modes = 100
    return cfg

# =========================================================================
#                       1. BEAM DATACLASS TESTS
# =========================================================================

def test_beam_properties_synthetic():
    """Manually construct a Beam to verify physical properties computation."""
    # 2 plane waves: 
    # Mode 0: travels along +z, w = 2*pi (lambda = 1.0)
    # Mode 1: travels along +y, w = 4*pi (lambda = 0.5)
    # Mode 2: travels along +x, w = 2*pi (lambda = 1.0)
    k = np.array([
        [0.0, 0.0, 2*np.pi],
        [0.0, 4*np.pi, 0.0],
        [2*np.pi, 0.0, 0.0]
    ])
    w = np.linalg.norm(k, axis=0)
    inv_w = 1.0 / w
    a = np.array([1.0 + 0j, 0.0 + 2.0j, 1.0 + 0j])
    c = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 2.0j, 0.0]
    ])
    
    beam = Beam(k=k, c=c, w=w, inv_w=inv_w, a=a)
    
    # Basic properties
    assert beam.num_modes == 3
    assert np.allclose(beam.wavelengths, [1.0, 0.5, 1.0])
    
    # k_hat (normalized directions)
    expected_k_hat = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    assert np.allclose(beam.k_hat, expected_k_hat)
    
    # Amplitudes & Irradiances
    assert np.allclose(beam.amplitudes, [1.0, 2.0, 1.0])
    assert np.allclose(beam.mode_irradiances, [1.0, 4.0, 1.0]) # |1|^2, |2j|^2, |1|^2
    assert beam.total_power == 6.0
    
    # mode_weights (1/6, 4/6, 1/6)
    assert np.allclose(beam.mode_weights, [1/6, 4/6, 1/6])
    
    # Polarizations (c / a)
    assert np.allclose(beam.polarizations[:, 0], [1, 0, 0])
    assert np.allclose(beam.polarizations[:, 1], [0, 0, 1])
    assert np.allclose(beam.polarizations[:, 2], [0, 1, 0])
    
    # Mean Direction (intensity weighted)
    expected_mean_dir = (
        np.array([0, 0, 1]) * (1/6) + 
        np.array([0, 1, 0]) * (4/6) + 
        np.array([1, 0, 0]) * (1/6)
    )
    expected_mean_dir /= np.linalg.norm(expected_mean_dir)
    assert np.allclose(beam.mean_direction, expected_mean_dir)

def test_beam_zero_power_fallback():
    """Tests the fallbacks when total power is virtually zero."""
    k = np.array([[0.0], [0.0], [1.0]])
    beam = Beam(
        k=k, 
        c=np.zeros((3, 1), dtype=complex),
        w=np.array([1.0]),
        inv_w=np.array([1.0]),
        a=np.array([0.0j])
    )
    
    assert beam.total_power == 0.0
    assert np.allclose(beam.mode_weights, [0.0])
    assert beam.rms_divergence == 0.0
    # Fallback for mean direction should pick the first mode
    assert np.allclose(beam.mean_direction, [0, 0, 1])
    
    # Empty spectrum fallback
    empty_beam = Beam(
        k=np.zeros((3, 0)), c=np.zeros((3, 0)), w=np.zeros(0), inv_w=np.zeros(0), a=np.zeros(0)
    )
    wls, spec = empty_beam.wavelength_spectrum
    assert len(wls) == 0

def test_beam_wavelength_spectrum():
    """Ensure it correctly groups identical wavelengths."""
    w = np.array([2*np.pi, 2*np.pi, 4*np.pi]) # Wavelengths: 1.0, 1.0, 0.5
    c = np.ones((3, 3)) # Irradiance = 3 per mode
    beam = Beam(
        k=np.zeros((3,3)), c=c, w=w, inv_w=1/w, a=np.zeros(3)
    )
    
    wls, spectra = beam.wavelength_spectrum
    assert len(wls) == 2
    # Sort order of wavelengths: 0.5, 1.0
    assert np.allclose(wls, [0.5, 1.0])
    # The two wl=1.0 modes should sum together (3.0 + 3.0 = 6.0)
    assert np.allclose(spectra, [3.0, 6.0])

def test_beam_plot_methods(monkeypatch):
    """Ensure plotting methods execute properly without raising logic errors."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    
    k = np.array([[0.0], [0.0], [2*np.pi]])
    beam = Beam(
        k=k, c=np.ones((3,1)), w=np.array([2*np.pi]), 
        inv_w=np.array([1/(2*np.pi)]), a=np.array([1.0])
    )
    
    beam.summary()
    beam.plot_k_perp_profile(show=False)
    beam.plot_wavelength_spectrum(show=False)
    beam.plot_kspace_3d(plot_type='matplotlib_scatter', show=False)
    
    try:
        import pyvista
        beam.plot_kspace_3d(plot_type='colored_vectors', show=False)
        beam.plot_kspace_3d(plot_type='colored_sphere', show=False)
        with pytest.raises(ValueError, match="plot_type must be"):
            beam.plot_kspace_3d(plot_type='invalid_type', show=False)
    except ImportError:
        pass


# =========================================================================
#                       2. BEAM MAKER TESTS
# =========================================================================

def test_beammaker_num_modes_errors(base_config):
    base_config.source.num_modes = 0
    with pytest.raises(ValueError, match="num_modes must be > 0"):
        BeamMaker(base_config).generate_beam()
        
    base_config.source.num_modes = 1
    with pytest.warns(UserWarning, match="pure single plane wave"):
        BeamMaker(base_config).generate_beam()
        
    base_config.source.num_modes = 5
    with pytest.warns(UserWarning, match="very low"):
        BeamMaker(base_config).generate_beam()

def test_beammaker_zero_power(base_config):
    """If profile always returns 0, beam has no power -> raise error."""
    def zero_profile(k): return 0.0 + 0j
    base_config.source.k_space.custom(zero_profile, vectorised=False)
    
    with pytest.raises(ValueError, match="essentially zero power"):
        BeamMaker(base_config).generate_beam()

def test_beammaker_clipping_warning(base_config):
    """Broad beam + narrow sampling cone should trigger clipping warning."""
    base_config.source.num_modes = 500
    base_config.source.theta_max = np.pi / 4  # Narrow cone
    # Gaussian with massive sigma to ensure it hits the edges hard
    base_config.source.k_space.gaussian(sigma_k_perp=100.0)
    
    with pytest.warns(UserWarning, match="Beam Clipping Detected"):
        BeamMaker(base_config).generate_beam()

def test_beammaker_polychromatic_splitting(base_config):
    """Tests correct generation and batch aggregation of multiple wavelengths."""
    base_config.source.wavelength = [0.4, 0.5, 0.6]
    base_config.source.num_modes = 150
    base_config.source.polychromatic.gaussian(center=0.5, sigma=0.1)
    
    beam = BeamMaker(base_config).generate_beam()
    assert beam.num_modes == 150
    wls, spec = beam.wavelength_spectrum
    assert len(wls) == 3
    
    # Check that the 0.5 wavelength has the highest spectral weight
    idx_center = np.argmin(np.abs(wls - 0.5))
    assert spec[idx_center] == np.max(spec)

def test_beammaker_zero_polychromatic_weights(base_config):
    """If polychromatic envelope evaluates to 0 everywhere, it should fallback evenly."""
    base_config.source.wavelength = [0.4, 0.5]
    def zero_poly(wl): return 0.0
    base_config.source.polychromatic.custom(zero_poly)
    
    beam = BeamMaker(base_config).generate_beam()
    wls, spec = beam.wavelength_spectrum
    # Both lines should have identical power fallback weight (1/sqrt(N))
    assert np.isclose(spec[0], spec[1], rtol=1e-3)

def test_beammaker_randomize_branches(base_config):
    """Tests execution of random noise generation algorithms."""
    base_config.source.num_modes = 50
    
    # 1. Test pol_state override and phase randomize
    base_config.source.randomize.pol_state = True
    base_config.source.randomize.phase_max = np.pi
    base_config.source.randomize.amplitude = True
    beam_pol = BeamMaker(base_config).generate_beam()
    assert beam_pol.total_power > 0
    
    # 2. Test normal pol randomization
    base_config.source.randomize.pol_state = False
    base_config.source.randomize.pol_rot_max = np.pi / 4
    beam_rot = BeamMaker(base_config).generate_beam()
    assert beam_rot.total_power > 0

def test_beammaker_non_vectorised_profile(base_config):
    """Tests the fallback branch for evaluating custom non-vectorised callables."""
    def scalar_profile(k): return 1.0 + 0j
    base_config.source.k_space.custom(scalar_profile, vectorised=False)
    beam = BeamMaker(base_config).generate_beam()
    assert beam.total_power > 0


def test_beammaker_axis_alignment(base_config):
    """Beam axis rotation preserves the expected mean propagation direction."""
    base_config.source.num_modes = 5000
    base_config.source.theta_max = 0.1

    # this isolates the rotation from the k-space profile.
    base_config.source.k_space.uniform()

    # Uniform k-space extends to theta_max, so clipping is expected here.
    with pytest.warns(UserWarning, match="Beam Clipping Detected"):
        base_config.source.beam_axis = (1, 0, 0)
        beam = BeamMaker(base_config).generate_beam()
        assert np.allclose(beam.mean_direction, [1, 0, 0], atol=1e-2)

        # Also covers the 180° (-z) rotation edge case.
        base_config.source.beam_axis = (0, 0, -1)
        beam = BeamMaker(base_config).generate_beam()
        assert np.allclose(beam.mean_direction, [0, 0, -1], atol=1e-2)

def test_beammaker_verbose(base_config, capsys):
    """Test verbose print statements."""
    base_config.verbose = True
    BeamMaker(base_config).generate_beam()
    captured = capsys.readouterr()
    assert "Starting Beam Generation" in captured.out