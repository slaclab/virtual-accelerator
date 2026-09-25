import time

import numpy as np
import pytest

from virtual_accelerator.tests.dependency_profiles import (
    HAS_BMAD_DEPS,
    HAS_LCLS_LATTICE,
)
from virtual_accelerator.models.special import get_cu_hxr_rmat


@pytest.mark.requires_bmad
@pytest.mark.requires_lcls_lattice
@pytest.mark.skipif(
    not HAS_BMAD_DEPS or not HAS_LCLS_LATTICE,
    reason="requires bmad optional dependencies and LCLS_LATTICE",
)
class TestGetCUHXRRmat:
    def test_rmat_variable_registered_and_read_only(self):
        model = get_cu_hxr_rmat("OTR2", "OTR4")

        rmat_name = "rmat:OTR2_OTR4"
        assert rmat_name in model.supported_variables
        assert model.supported_variables[rmat_name].read_only is True

    def test_rmat_shape_and_dtype(self):
        model = get_cu_hxr_rmat("WS27644", "WS28144")

        rmat = model.get_value("rmat:WS27644_WS28144")
        assert isinstance(rmat, np.ndarray)
        assert rmat.shape == (6, 6)
        assert rmat.dtype == float

    def test_rmat_matches_direct_tao_calculation(self):
        from lume_bmad.utils import rmat_get

        model = get_cu_hxr_rmat("OTR2", "OTR4")

        rmat = model.get_value("rmat:OTR2_OTR4")
        expected = rmat_get(model.tao, "OTR2", "OTR4")
        assert np.allclose(rmat, expected)

    def test_rmat_does_not_enable_beam_tracking(self):
        model = get_cu_hxr_rmat("OTR2", "OTR4")

        assert model.tao.tao_global()["track_type"] != "beam"

    def test_rmat_updates_after_control_variable_change(self):
        model = get_cu_hxr_rmat("OTR2", "OTR4")

        control_variable = next(
            name
            for name, variable in model.supported_variables.items()
            if not getattr(variable, "read_only", True)
        )

        initial_rmat = model.get_value("rmat:OTR2_OTR4").copy()
        model.set({control_variable: model.get_value(control_variable) + 1e-3})
        updated_rmat = model.get_value("rmat:OTR2_OTR4")

        assert not np.allclose(initial_rmat, updated_rmat)

    def test_rmat_get_value_performance(self):
        model = get_cu_hxr_rmat("WS27644", "WS28144")

        # warm up so lazy imports/caching don't skew the timing
        model.set({})
        model.get_value("rmat:WS27644_WS28144")

        n_calls = 20
        start = time.perf_counter()
        for _ in range(n_calls):
            model.set({})
            model.get_value("rmat:WS27644_WS28144")
        elapsed = time.perf_counter() - start

        mean_time = elapsed / n_calls
        print(f"rmat get_value mean time: {mean_time * 1e3:.3f} ms")

        assert mean_time < 100e-3
