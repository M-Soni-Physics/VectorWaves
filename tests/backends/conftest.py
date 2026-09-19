import pytest
import numpy as np
from vectorwaves.beam_stuff import Beam

@pytest.fixture
def simple_test_beam():
    """
    Creates a deterministic 2-mode Beam for testing physical backends.
    - Mode 0: Propagates along +z, polarized along x. lambda = 1.0 -> w = 2*pi
    - Mode 1: Propagates along +y, polarized along z. lambda = 2.0 -> w = pi
    """
    k = np.array([
        [0.0, 0.0],          # kx
        [0.0, np.pi],        # ky
        [2.0 * np.pi, 0.0]   # kz
    ])
    w = np.array([2.0 * np.pi, np.pi])
    inv_w = 1.0 / w
    
    # Electric field complex amplitudes c
    c = np.array([
        [1.0 + 0j, 0.0 + 0j],  # cx
        [0.0 + 0j, 0.0 + 0j],  # cy
        [0.0 + 0j, 2.0 + 0j]   # cz
    ])
    a = np.array([1.0, 2.0]) # scalar amplitude amplitudes
    
    return Beam(k=k, c=c, w=w, inv_w=inv_w, a=a)