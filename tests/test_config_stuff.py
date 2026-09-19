import os
import json
import pytest
import numpy as np
from pathlib import Path

from vectorwaves.config_stuff import (
    get_config, load_config, Config, OpConfig, RandomizeConfig, 
    KSpaceConfig, PolychromaticConfig, SourceConfig, SerializableConfig
)
from vectorwaves.spectra import KSpaceSpectra, PolychromaticSpectra

# =========================================================================
#                       1. OP CONFIG TESTS
# =========================================================================

def test_op_config_valid():
    op = OpConfig(spacing=0.1, size=(5.0, 5.0), center=[1.0, -1.0])
    assert op.spacing == 0.1
    assert op.size == (5.0, 5.0)  # Check tuple coercion
    assert op.center == (1.0, -1.0)

def test_op_config_physical_limits():
    with pytest.raises(ValueError, match="must be > 0"):
        OpConfig(spacing=-0.1)
    
    with pytest.raises(ValueError, match="op.size must be positive"):
        OpConfig(size=(0, 10))

def test_op_config_type_coercion():
    with pytest.raises(TypeError, match="Cannot convert"):
        OpConfig(spacing="not_a_number")
    
    with pytest.raises(ValueError, match="Expected 2 elements"):
        OpConfig(size=(10.0,))


# =========================================================================
#                       2. RANDOMIZE CONFIG TESTS
# =========================================================================

def test_randomize_config_bounds():
    RandomizeConfig(phase_max=0, pol_rot_max=np.pi)
    
    with pytest.raises(ValueError, match="must be in"):
        RandomizeConfig(phase_max=np.pi + 0.1)

    with pytest.raises(ValueError, match="must be in"):
        RandomizeConfig(pol_rot_max=-0.1)

def test_randomize_type_errors():
    with pytest.raises(TypeError, match="Expected bool"):
        RandomizeConfig(pol_state="not_a_bool")
    
    with pytest.raises(TypeError, match="Expected bool"):
        RandomizeConfig(amplitude=123)

def test_randomize_override_warning():
    with pytest.warns(UserWarning, match="overrides"):
        RandomizeConfig(pol_state=True, pol_rot_max=1.0)

def test_randomize_off_method():
    rnd = RandomizeConfig().off()
    assert rnd.pol_rot_max == 0
    assert rnd.phase_max == 0
    assert rnd.pol_state is False
    assert rnd.amplitude is False


# =========================================================================
#                       3. K-SPACE CONFIG TESTS
# =========================================================================

def test_kspace_fluent_api():
    k = KSpaceConfig()
    
    k.gaussian(sigma_k_perp=3.5)
    assert k.profile == KSpaceSpectra.gaussian
    assert k.params['sigma_k_perp'] == 3.5
    
    k.laguerre_gauss(p=1, l=2)
    assert k.params['p'] == 1 and k.params['l'] == 2

    with pytest.raises(ValueError, match="must be >= 0"):
        k.laguerre_gauss(p=-1)
        
    with pytest.raises(ValueError, match="HG indices l, m must be >= 0"):
        k.hermite_gauss(l=-1, m=0)

def test_kspace_bessel_warning():
    k = KSpaceConfig()
    with pytest.warns(UserWarning, match="Large Bessel cone angles"):
        k.bessel_gauss(theta_deg=75.0)

def test_kspace_custom_callable():
    k = KSpaceConfig()
    
    # Valid non-vectorized
    def my_valid_profile(k_vec, a=1): return 1.0 + 0j
    k.custom(my_valid_profile, vectorised=False, a=2)
    assert k.params['a'] == 2

    # Invalid non-vectorized (returns string)
    def my_invalid_profile(k_vec): return "hello"
    with pytest.raises(RuntimeError, match="must return a scalar"):
        k.custom(my_invalid_profile, vectorised=False)

    # Invalid vectorized (returns scalar instead of array)
    def my_invalid_vec_profile(k_arr): return 1.0
    with pytest.raises(RuntimeError, match="must return an array matching input shape"):
        k.custom(my_invalid_vec_profile, vectorised=True)

def test_kspace_custom_callable_exception():
    k = KSpaceConfig()
    def crashing_profile(k_vec): raise ArithmeticError("Math failed!")
    with pytest.raises(RuntimeError, match="K space profile failed validation: Math failed!"):
        k.custom(crashing_profile)

def test_kspace_custom_callable_valid_vectorized():
    k = KSpaceConfig()
    # Must accept (3, N) and return (N,) array
    def valid_vec(k_arr): return np.ones(k_arr.shape[1], dtype=complex)
    k.custom(valid_vec, vectorised=True)
    assert k.vectorised is True

# =========================================================================
#                       4. POLYCHROMATIC CONFIG TESTS
# =========================================================================

def test_polychromatic_fluent_api():
    p = PolychromaticConfig()
    p.lorentzian(center=5.0, gamma=1.0)
    assert p.profile == PolychromaticSpectra.lorentzian
    assert p.params['center'] == 5.0
    
def test_polychromatic_custom_callable():
    p = PolychromaticConfig()
    
    def bad_envelope(wl): return [1, 2] # Must be scalar
    with pytest.raises(RuntimeError, match="must return a scalar"):
        p.custom(bad_envelope)

    def inf_envelope(wl): return np.inf # Must be finite
    with pytest.raises(RuntimeError, match="non-finite"):
        p.custom(inf_envelope)

def test_polychromatic_custom_callable_valid():
    p = PolychromaticConfig()
    def valid_env(wl): return float(wl * 2.0)
    p.custom(valid_env)
    assert p.profile(8.0) == 16.0

# =========================================================================
#                       5. SOURCE CONFIG TESTS
# =========================================================================

def test_source_vector_normalization():
    src = SourceConfig(beam_axis=(0, 2, 0), pol_vect=(1j, 1j))
    assert np.allclose(src.beam_axis, (0, 1, 0))
    expected_pol = (1j / np.sqrt(2), 1j / np.sqrt(2))
    assert np.isclose(src.pol_vect[0], expected_pol[0])
    assert np.isclose(src.pol_vect[1], expected_pol[1])

def test_source_zero_vectors():
    with pytest.raises(ValueError, match="reasonable norm"):
        SourceConfig(beam_axis=(0, 0, 0))
        
    with pytest.raises(ValueError, match="reasonable norm"):
        SourceConfig(pol_vect=(0, 0))

def test_source_wavelength_array():
    src = SourceConfig(wavelength=[0.5, 0.6, 0.7])
    assert isinstance(src.wavelength, list)
    assert src.wavelength == [0.5, 0.6, 0.7]

    with pytest.raises(ValueError, match="All wavelengths must be > 0"):
        SourceConfig(wavelength=[0.5, -0.6])
        
def test_source_config_limits_and_types():
    with pytest.raises(ValueError, match="intensity_scale must be > 0"):
        SourceConfig(intensity_scale=-1.0)
        
    with pytest.raises(ValueError, match="num_modes must be > 0"):
        SourceConfig(num_modes=0)
        
    with pytest.raises(TypeError, match="Expected float or list of floats"):
        SourceConfig(wavelength={"val": 0.5})
        
    with pytest.raises(ValueError, match="maximum angle must be in"):
        SourceConfig(theta_max=np.pi + 0.1)

def test_source_wavelength_ndarray():
    # Source config converts iterables to lists of floats internally
    src = SourceConfig(wavelength=np.array([0.5, 0.6]))
    assert isinstance(src.wavelength, list)
    assert src.wavelength == [0.5, 0.6]

def test_source_theta_max_zero_or_negative():
    with pytest.raises(ValueError, match="maximum angle must be in"):
        SourceConfig(theta_max=0.0)
    with pytest.raises(ValueError, match="maximum angle must be in"):
        SourceConfig(theta_max=-1.0)

# =========================================================================
#                       6. MAIN CONFIG TESTS
# =========================================================================

def test_config_backends():
    Config(backend="NUMBA") # Should coerce to lowercase
    
    with pytest.raises(ValueError, match="Invalid backend"):
        Config(backend="pytorch")

def test_config_aliasing_warning():
    with pytest.warns(UserWarning, match="Spatial aliasing detected"):
        Config(op=OpConfig(spacing=0.05), source=SourceConfig(wavelength=0.05))


# =========================================================================
#                       7. SERIALIZATION (SAVE/LOAD)
# =========================================================================

def test_serialization_roundtrip(tmp_path):
    c1 = get_config()
    c1.op.spacing = 0.0123
    c1.source.pol_vect = (1, 0.5j)
    c1.source.k_space.hermite_gauss(l=1, m=1)
    c1.source.wavelength = [0.4, 0.5, 0.6]
    c1.validate()

    filepath = str(tmp_path / "config_roundtrip.json")
    c1.save(filepath)
    c2 = load_config(filepath)

    assert c2.op.spacing == pytest.approx(c1.op.spacing, rel=1e-9, abs=1e-12)
    assert c2.source.pol_vect[0].real == pytest.approx(c1.source.pol_vect[0].real)
    assert c2.source.pol_vect[0].imag == pytest.approx(c1.source.pol_vect[0].imag)
    assert c2.source.pol_vect[1].real == pytest.approx(c1.source.pol_vect[1].real)
    assert c2.source.pol_vect[1].imag == pytest.approx(c1.source.pol_vect[1].imag)
    assert c2.source.wavelength == pytest.approx(c1.source.wavelength, rel=1e-9)
    assert c2.source.k_space.profile.__name__ == 'hermite_gauss'

def test_save_requires_json_extension():
    c = get_config()
    with pytest.raises(ValueError, match="must end with '.json'"):
        c.save("my_config.txt")

def test_custom_callable_serialization_warning(tmp_path):
    c = get_config()
    
    def my_custom_k(k): return 1.0
    c.source.k_space.custom(my_custom_k)
    
    filepath = str(tmp_path / "custom_config.json")
    c.save(filepath)
    
    with pytest.warns(UserWarning, match="Custom callable"):
        loaded_c = load_config(filepath)
        
    with pytest.raises(RuntimeError, match="Custom callable.*not initialized properly"):
        loaded_c.source.k_space.profile()

def test_deserialize_callable_invalid():
    # If the JSON file was hacked/modified and points to a non-existent builtin
    data = {
        "__callable__": True, 
        "type": "builtin", 
        "class": "KSpaceSpectra", 
        "name": "non_existent_method_123"
    }
    with pytest.raises(RuntimeError, match="Deserializing callable failed"):
        SerializableConfig._deserialize_callable(data)

def test_save_enforces_validation_when_mutated(tmp_path):
    c = get_config()
    c.op.spacing = -5.0  # User manually breaks the state
    
    filepath = str(tmp_path / "bad_state.json")
    with pytest.raises(ValueError, match="op.spacing must be > 0"):
        c.save(filepath) # Should crash before writing anything!

# =========================================================================
#                       8. PROFILE PARAMETRIZATIONS
# =========================================================================

@pytest.mark.parametrize("method_name, kwargs, expected_params", [
    ("uniform", {}, {}),
    ("gaussian", {"sigma_k_perp": 2.5}, {"sigma_k_perp": 2.5}),
    ("tophat", {"k_perp_max": 2.0}, {"k_perp_max": 2.0}),
    ("laguerre_gauss", {"p": 2, "l": 3}, {"p": 2, "l": 3, "sigma_k_perp": 0.5}),
    ("hermite_gauss", {"l": 1, "m": 1}, {"l": 1, "m": 1, "sigma_k_perp": 0.5}),
    ("bessel_gauss", {"theta_deg": 10.0}, {"theta_0": np.radians(10.0), "sigma_theta": 0.05, "l": 0}),
])
def test_kspace_all_profiles(method_name, kwargs, expected_params):
    k = KSpaceConfig()
    getattr(k, method_name)(**kwargs)
    assert k.profile.__name__ == method_name
    assert k.params == expected_params
    assert k.vectorised is True


@pytest.mark.parametrize("method_name, kwargs", [
    ("uniform", {}),
    ("gaussian", {"center": 5.0, "sigma": 0.1}),
    ("lorentzian", {"center": 5.0, "gamma": 0.1}),
    ("tophat", {"center": 5.0, "width": 1.0}),
])
def test_polychromatic_all_profiles(method_name, kwargs):
    p = PolychromaticConfig()
    getattr(p, method_name)(**kwargs)
    assert p.profile.__name__ == method_name
    for key, val in kwargs.items():
        assert p.params[key] == val

def test_migration_stub():
    data = {"__version__": (0, 0, 1), "op": {}}
    with pytest.raises(ValueError, match="Unsupported config version"):
        Config.from_dict(data)

def test_unknown_kwargs_in_json():
    data = {"op": {}, "fake_parameter": 123}
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        Config.from_dict(data)

def test_internal_helpers_edge_cases():
    from vectorwaves.config_stuff import _check_scalar, _coerce_tuple
    
    assert _check_scalar(5, "test", float) == 5.0
    
    with pytest.raises(ValueError, match="finite number"):
        _check_scalar(np.inf, "test", float)
        
    with pytest.raises(TypeError, match="Expected float, got complex"):
        _check_scalar(1+2j, "test", float, allow_complex=False)
        
    with pytest.raises(TypeError, match="Expected strict integer"):
        _check_scalar(5.5, "test", int)
        
    res = _coerce_tuple([1, 2, 3], 3, "test", float)
    assert res == (1.0, 2.0, 3.0)
    
    with pytest.raises(TypeError, match="Expected a sequence"):
        _coerce_tuple("123", 3, "test")