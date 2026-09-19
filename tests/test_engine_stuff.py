import pytest
import numpy as np
from vectorwaves.config_stuff import get_config
from vectorwaves.beam_stuff import Beam
from vectorwaves.engine_stuff import FieldEngine, FieldResult

@pytest.fixture
def test_beam():
    """Provides a simple 1-mode Beam propagating along +z with x-polarization."""
    k = np.array([[0.0], [0.0], [2 * np.pi]])
    w = np.array([2 * np.pi])
    inv_w = 1.0 / w
    c = np.array([[1.0 + 0j], [0.0 + 0j], [0.0 + 0j]])
    a = np.array([1.0])
    return Beam(k=k, c=c, w=w, inv_w=inv_w, a=a)

@pytest.fixture
def engine_config():
    """Provides a deterministic configuration for engine testing."""
    cfg = get_config()
    cfg.op.spacing = 0.5
    cfg.op.size = (2.0, 2.0)
    cfg.op.center = (0.0, 0.0)
    cfg.source.num_modes = 10
    return cfg

# =========================================================================
#                    1. FIELDRESULT TESTS
# =========================================================================

def test_field_result_basic_and_exceptions():
    # Instantiation
    E = np.ones((3, 4, 4), dtype=complex)
    res = FieldResult(E=E)
    
    assert repr(res) == "<FieldResult: Grid=(4, 4), B_computed=False, Derivs_computed=False>"
    
    with pytest.raises(RuntimeError, match="Magnetic field.*not computed"):
        _ = res.B
        
    with pytest.raises(RuntimeError, match="Derivatives were not computed"):
        _ = res.jacobian_E

def test_field_result_calculus_properties():
    # E shape: (3, 2, 2)
    E = np.ones((3, 2, 2), dtype=complex)
    
    # Jacobian shape: (3, 3, 2, 2)
    # Numerator convention: jac[i, j] = dE_i / dx_j
    jac = np.zeros((3, 3, 2, 2), dtype=complex)
    # Set dEx_dx = 1.0, dEy_dy = 2.0, dEz_dz = 3.0 -> div_E should be 6.0
    jac[0, 0, :, :] = 1.0
    jac[1, 1, :, :] = 2.0
    jac[2, 2, :, :] = 3.0
    
    # Set dEz_dy = 4.0, dEy_dz = 1.0 -> curl_E[0] = 3.0 (4.0 - 1.0)
    jac[2, 1, :, :] = 4.0
    jac[1, 2, :, :] = 1.0
    
    # Set dEx_dz = 5.0, dEz_dx = 2.0 -> curl_E[1] = 3.0 (5.0 - 2.0)
    jac[0, 2, :, :] = 5.0
    jac[2, 0, :, :] = 2.0
    
    # Set dEy_dx = 6.0, dEx_dy = 3.0 -> curl_E[2] = 3.0 (6.0 - 3.0)
    jac[1, 0, :, :] = 6.0
    jac[0, 1, :, :] = 3.0
    
    res = FieldResult(E=E, _jacobian_E=jac)
    
    assert np.allclose(res.div_E, 6.0)
    assert np.allclose(res.curl_E, 3.0)
    assert np.allclose(res.dE_dx, jac[:, 0, :, :])
    assert np.allclose(res.dE_dy, jac[:, 1, :, :])
    assert np.allclose(res.dE_dz, jac[:, 2, :, :])
    assert np.allclose(res.intensity_E, 3.0)

# =========================================================================
#                    2. FIELDENGINE TESTS
# =========================================================================

def test_field_engine_initialization(test_beam, engine_config):
    engine = FieldEngine(test_beam, engine_config)
    
    # Size: (2.0, 2.0) with spacing 0.5 -> nx = 4, ny = 4
    assert len(engine.x) == 4
    assert len(engine.y) == 4
    assert engine.X.shape == (4, 4)
    assert engine.Y.shape == (4, 4)
    assert np.allclose(engine.op_extent, [-1.0, 1.0, -1.0, 1.0])

def test_field_engine_selector_routing(test_beam, engine_config):
    engine = FieldEngine(test_beam, engine_config)
    
    # Explicit Numpy choice
    assert engine.selector("numpy") == "numpy"
    
    # Case insensitivity
    assert engine.selector("NuMpY") == "numpy"
    
    # Unsupported selection
    with pytest.raises(ValueError, match="not supported"):
        engine.selector("invalid_backend_name")

    # If CuPy is missing, requesting it must fail
    from vectorwaves.backends.cupy_backend import has_cupy
    if not has_cupy:
        with pytest.raises(ValueError, match="CuPy backend requested but cupy is not installed"):
            engine.selector("cupy32")

def test_field_engine_compute_on_op(test_beam, engine_config):
    engine = FieldEngine(test_beam, engine_config)
    
    res = engine.compute_on_op(z=0.0, t=0.1, need_b=True, need_derivs=True, backend_name="numpy")
    
    assert res.E.shape == (3, 4, 4)
    assert res.B.shape == (3, 4, 4)
    assert res.jacobian_E.shape == (3, 3, 4, 4)
    assert repr(res) == "<FieldResult: Grid=(4, 4), B_computed=True, Derivs_computed=True>"
    
    # Scalar constraint checks
    with pytest.raises(ValueError, match="requires 'z' and 't' to be scalars"):
        engine.compute_on_op(z=[0.0], t=0.0)

def test_field_engine_compute_cloud(test_beam, engine_config):
    engine = FieldEngine(test_beam, engine_config)
    
    x = np.array([0.0, 0.5])
    y = np.array([0.0, -0.5])
    z = np.array([1.0, 2.0])
    
    res = engine.compute_cloud(x, y, z, t=0.0, need_b=True, need_derivs=True, backend_name="numpy")
    assert res.E.shape == (3, 2)
    assert res.B.shape == (3, 2)
    assert res.jacobian_E.shape == (3, 3, 2)
    assert repr(res) == "<FieldResult: Grid=(2,), B_computed=True, Derivs_computed=True>"
    
    # Dimension checking failures
    with pytest.raises(ValueError, match="requires 1D arrays"):
        engine.compute_cloud(x.reshape(2, 1), y, z)
        
    with pytest.raises(ValueError, match="identical length"):
        engine.compute_cloud(x, y, np.array([1.0]))
        
    with pytest.raises(ValueError, match="must be a scalar"):
        engine.compute_cloud(x, y, z, t=np.array([1.0, 2.0]))

def test_field_engine_compute_point(test_beam, engine_config):
    engine = FieldEngine(test_beam, engine_config)
    
    res = engine.compute_point(0.1, 0.2, 0.3, t=0.0, need_b=True, need_derivs=True, backend_name="numpy")
    assert res.E.shape == (3,)
    assert res.B.shape == (3,)
    assert res.jacobian_E.shape == (3, 3)
    assert repr(res) == "<FieldResult: Grid=Point, B_computed=True, Derivs_computed=True>"
    
    # Non-scalar input failures
    with pytest.raises(ValueError, match="requires pure scalars"):
        engine.compute_point([0.1], 0.2, 0.3)


def test_field_engine_various_backends_and_progress(test_beam, engine_config):
    """Exercises JIT execution, backend selection, and progress bar hooks."""
    engine = FieldEngine(test_beam, engine_config)
    
    # 1. Numba backend evaluation + progress meter
    res_numba = engine.compute_on_op(
        z=0.0, t=0.1, need_b=True, need_derivs=True, 
        backend_name="numba", progress_bar=True
    )
    assert res_numba.E.shape == (3, 4, 4)
    
    # 2. Automatic backend routing (triggers self.backend_name initialization flow)
    res_auto = engine.compute_on_op(
        z=0.0, t=0.1, need_b=True, need_derivs=True, 
        backend_name="auto"
    )
    assert res_auto.E.shape == (3, 4, 4)
    
    # 3. Cloud evaluation + progress meter
    x = np.array([0.0, 0.5])
    y = np.array([0.0, -0.5])
    z = np.array([1.0, 2.0])
    res_cloud = engine.compute_cloud(
        x, y, z, t=0.0, need_b=True, need_derivs=True, 
        backend_name="numba", progress_bar=True
    )
    assert res_cloud.E.shape == (3, 2)