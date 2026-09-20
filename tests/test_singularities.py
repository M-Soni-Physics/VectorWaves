import pytest
import numpy as np
import warnings
from vectorwaves.engine_stuff import FieldResult
from vectorwaves.singularities import SingularityFinder

# =========================================================================
#                    1. COORDINATE-MAPPED ANALYTICAL MOCK ENGINE
# =========================================================================

class MockFieldResult:
    """Mock FieldResult carrying analytic components and Jacobians."""
    def __init__(self, E, J):
        self.E = E
        self.jacobian_E = J

class MockFieldEngine:
    """
    Simulates a FieldEngine evaluating fields on an Observation Plane or Cloud.
    Models simple isolated singularities at the coordinate trajectory:
       x = z, y = z
    This creates an exact 3D line trajectory of singularities propagating along z.
    """
    def __init__(self):
        # Use an even number of grid points to avoid exact coordinate 0.0,
        # preventing np.sign(0.0) == 0 from masking zero-crossing cells.
        self.x = np.linspace(-0.5, 0.5, 10)
        self.y = np.linspace(-0.5, 0.5, 10)
        self.backend_name = 'numba'
        self.mode = 'C'
        
    def compute_cloud(self, x_arr, y_arr, z_arr, t=0.0, need_b=False):
        N = len(x_arr)
        E = np.zeros((3, N), dtype=complex)
        J = np.zeros((3, 3, N), dtype=complex)
        
        # Singularity location shifts linearly with z
        x_shifted = x_arr - z_arr
        y_shifted = y_arr - z_arr
        
        if self.mode == 'C':
            Ex = np.sqrt(1.5 + y_shifted) * np.exp(1j * x_shifted)
            Ey = 1j * np.sqrt(1.5 - y_shifted)
            E[0] = Ex
            E[1] = Ey
            
            # Gradients
            J[0, 0] = 1j * Ex                      # dEx/dx
            J[0, 1] = 0.5 * Ex / (1.5 + y_shifted)  # dEx/dy
            J[1, 1] = -0.5 * Ey / (1.5 - y_shifted) # dEy/dy
            
        elif self.mode == 'CT':
            # E . E = 0 at (0, 0)
            # Add spatial dependency to Ex and Ez so their gradients at the 
            # singularity are non-zero, ensuring a non-zero tangent vector.
            Ex = 1.0 + x_shifted
            Ey = x_shifted + 1j * y_shifted
            Ez = 1j + y_shifted
            
            E[0] = Ex
            E[1] = Ey
            E[2] = Ez
            
            J[0, 0] = 1.0  # dEx/dx
            J[1, 0] = 1.0  # dEy/dx
            J[1, 1] = 1j   # dEy/dy
            J[2, 1] = 1.0  # dEz/dy
            
        elif self.mode == 'LT':
            E[0] = 1.0 + 1j
            E[1] = 1j * x_shifted
            E[2] = y_shifted
            
            J[1, 0] = 1j   # dEy/dx
            J[2, 1] = 1.0  # dEz/dy
            
        # Standardize z-derivatives under linear coordinate shift
        # d/dz f(x - z, y - z) = -d/dx f - d/dy f
        J[:, 2, :] = - J[:, 0, :] - J[:, 1, :]
            
        return MockFieldResult(E=E, J=J)


# =========================================================================
#                    2. INITIALIZATION & WARNING TESTS
# =========================================================================

def test_singularity_finder_backend_warnings():
    engine = MockFieldEngine()
    
    # Numpy warning trigger test
    engine.backend_name = 'numpy'
    with pytest.warns(RuntimeWarning, match="numpy.*backend"):
        SingularityFinder(engine)
        
    # cupy32 unsupported error trigger test
    engine.backend_name = 'cupy32'
    with pytest.raises(RuntimeError, match="cupy32 backend is not supported"):
        SingularityFinder(engine)


# =========================================================================
#                    3. 2D SINGULARITY LOCATOR TESTS
# =========================================================================

def test_find_stokes_C_points():
    engine = MockFieldEngine()
    engine.mode = 'C'
    finder = SingularityFinder(engine)
    
    # Grid plane setup at z=0
    X, Y = np.meshgrid(engine.x, engine.y)
    Ex = np.sqrt(1.5 + Y) * np.exp(1j * X)
    Ey = 1j * np.sqrt(1.5 - Y)
    
    E_grid = np.zeros((3, len(engine.y), len(engine.x)), dtype=complex)
    E_grid[0] = Ex
    E_grid[1] = Ey
    
    pts = finder.find_stokes_C_points(z_value=0.0, E_grid=E_grid)
    
    assert len(pts) > 0
    # Refined root position must converge close to (0.0, 0.0, 0.0)
    best_pt = min(pts, key=lambda p: np.hypot(p['position'][0], p['position'][1]))
    assert np.allclose(best_pt['position'][:2], [0.0, 0.0], atol=1e-5)
    assert best_pt['confident'] is True
    assert best_pt['type'] in ['Star', 'Lemon', 'Monstar']


def test_find_C_T_points():
    engine = MockFieldEngine()
    engine.mode = 'CT'
    finder = SingularityFinder(engine)
    
    X, Y = np.meshgrid(engine.x, engine.y)
    E_grid = np.zeros((3, len(engine.y), len(engine.x)), dtype=complex)
    E_grid[0] = 1.0 + X
    E_grid[1] = X + 1j * Y
    E_grid[2] = 1j + Y
    
    pts = finder.find_C_T_points(z_value=0.0, E_grid=E_grid)
    
    assert len(pts) > 0
    best_pt = min(pts, key=lambda p: np.hypot(p['position'][0], p['position'][1]))
    assert np.allclose(best_pt['position'][:2], [0.0, 0.0], atol=1e-5)
    assert best_pt['confident'] is True


def test_find_L_T_points():
    engine = MockFieldEngine()
    engine.mode = 'LT'
    finder = SingularityFinder(engine)
    
    X, Y = np.meshgrid(engine.x, engine.y)
    E_grid = np.zeros((3, len(engine.y), len(engine.x)), dtype=complex)
    E_grid[0] = 1.0 + 1j
    E_grid[1] = 1j * X
    E_grid[2] = Y
    
    pts = finder.find_L_T_points(z_value=0.0, E_grid=E_grid)
    
    assert len(pts) > 0
    best_pt = min(pts, key=lambda p: np.hypot(p['position'][0], p['position'][1]))
    assert np.allclose(best_pt['position'][:2], [0.0, 0.0], atol=1e-5)
    assert best_pt['confident'] is True


def test_empty_candidates_handling():
    engine = MockFieldEngine()
    finder = SingularityFinder(engine)
    
    # Evaluating a uniform field containing no crossings of any kind
    E_grid = np.ones((3, len(engine.y), len(engine.x)), dtype=complex)
    
    assert len(finder.find_stokes_C_points(0.0, E_grid)) == 0
    assert len(finder.find_C_T_points(0.0, E_grid)) == 0
    assert len(finder.find_L_T_points(0.0, E_grid)) == 0


def test_singular_determinant_handling():
    engine = MockFieldEngine()
    finder = SingularityFinder(engine)
    
    # Singular plane inputs (completely flat planes with determinant = 0)
    candidate_coords = np.array([[2, 2]])
    data = np.zeros((10, 10))
    
    # Suppress the expected NumPy division-by-zero/NaN warnings during the singular checks
    with np.errstate(divide='ignore', invalid='ignore'):
        _, _, valid_2eq = finder._batched_plane_fit_2eq(candidate_coords, data, data)
        assert not valid_2eq[0]
        
        _, _, valid_3eq = finder._batched_plane_fit_3eq(candidate_coords, data, data, data)
        assert not valid_3eq[0]

# =========================================================================
#                    4. 3D LINE TRACING TESTS
# =========================================================================

def test_trace_stokes_C_lines():
    engine = MockFieldEngine()
    engine.mode = 'C'
    finder = SingularityFinder(engine)
    
    starting_points = [(0.0, 0.0, 0.0)]
    # Walk along z coordinate, ds = 0.05, trace 5 steps
    trajectories = finder.trace_stokes_C_lines(starting_points, ds=0.05, max_steps=5)
    
    assert len(trajectories) == 1
    traj = trajectories[0]
    assert traj.shape[0] > 1
    for point in traj:
        x, y, z = point
        # The analytical singularity traces along the shifted diagonal: x = z, y = z
        assert np.isclose(x, z, atol=1e-3)
        assert np.isclose(y, z, atol=1e-3)

@pytest.mark.parametrize("seeds", [
    [(0.0, 0.0, 0.0), (0.02, 0.0, 0.0)],
    [(0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.2, 0.2, 0.2)],
    [(0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.2, 0.2, 0.2), (-0.2, -0.2, -0.2)],
])
def test_trace_stokes_C_lines_multiple(seeds):
    """Batched tracing with staggered corrector convergence (regression for M > 1)."""
    engine = MockFieldEngine()
    engine.mode = 'C'
    finder = SingularityFinder(engine)

    trajectories = finder.trace_stokes_C_lines(seeds, ds=0.05, max_steps=5)

    assert len(trajectories) == len(seeds)
    for traj in trajectories:
        assert traj.shape[0] > 1
        # Skip traj[0]: an off-line seed is stored as given, before any correction.
        for x, y, z in traj[1:]:
            assert np.isclose(x, z, atol=1e-3)
            assert np.isclose(y, z, atol=1e-3)

def test_trace_C_T_lines():
    engine = MockFieldEngine()
    engine.mode = 'CT'
    finder = SingularityFinder(engine)
    
    starting_points = [(0.0, 0.0, 0.0)]
    trajectories = finder.trace_C_T_lines(starting_points, ds=0.05, max_steps=5)
    
    assert len(trajectories) == 1
    traj = trajectories[0]
    assert traj.shape[0] > 1
    for point in traj:
        x, y, z = point
        assert np.isclose(x, z, atol=1e-3)
        assert np.isclose(y, z, atol=1e-3)


def test_trace_L_lines():
    engine = MockFieldEngine()
    engine.mode = 'LT'
    finder = SingularityFinder(engine)
    
    starting_points = [(0.0, 0.0, 0.0)]
    trajectories = finder.trace_L_lines(starting_points, ds=0.05, max_steps=5)
    
    assert len(trajectories) == 1
    traj = trajectories[0]
    assert traj.shape[0] > 1
    for point in traj:
        x, y, z = point
        assert np.isclose(x, z, atol=1e-3)
        assert np.isclose(y, z, atol=1e-3)


def test_empty_starting_points():
    engine = MockFieldEngine()
    finder = SingularityFinder(engine)
    assert finder.trace_stokes_C_lines([]) == []

