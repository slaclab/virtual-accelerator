import os
import importlib.util
from pathlib import Path
import pytest
from beamphysics.particles import ParticleGroup

# Guard collection: lume_torch is required by InjectorSurrogate at import
# time; skip the whole module instead of raising an ImportError.
pytest.importorskip(
    "lume_torch",
    reason="requires lume-torch: pip install virtual-accelerator[surrogate]",
)
pytest.importorskip(
    "facet2_inj_ml_model",
    reason="requires facet2_inj_ml_model: pip install virtual-accelerator[surrogate]",
)
pytest.importorskip(
    "lcls_cu_inj_model",
    reason="requires packaged Cu injector model: pip install virtual-accelerator[surrogate]",
)

from virtual_accelerator.models.staged_model import (  # noqa: E402
    StagedModel,
    get_cu_hxr_staged_model,
)
from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model  # noqa: E402
from virtual_accelerator.models.facet2 import get_facet_staged_model  # noqa: E402
from virtual_accelerator.surrogates.injector_surrogate import InjectorSurrogate  # noqa: E402

TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "../bmad", "test_beam")


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


pytestmark = pytest.mark.skipif(
    not all(
        _has_module(module_name)
        for module_name in (
            "pytao",
            "lume_bmad",
            "cheetah",
            "lume_cheetah",
            "lume_torch",
            "facet2_inj_ml_model",
        )
    ),
    reason="requires staged-model optional dependencies",
)


# Fixtures for model initialization
@pytest.fixture
def injector_model():
    """Fixture providing an InjectorSurrogate instance."""
    return InjectorSurrogate()


@pytest.fixture
def cu_hxr_bmad_model():
    """Fixture providing a cu_hxr BMAD model with custom beam path."""
    return get_cu_hxr_bmad_model(
        custom_beam_path=TEST_BEAM_PATH, end_element="OTR4", track_beam=True
    )


@pytest.fixture
def staged_model(injector_model, cu_hxr_bmad_model):
    """Fixture providing a basic StagedModel combining injector and cu_hxr."""
    return StagedModel([injector_model, cu_hxr_bmad_model])


class TestStagedModelValidation:
    """Test StagedModel validation and initialization."""

    def test_staged_model_initialization(self, staged_model):
        """Test valid StagedModel initialization."""
        assert len(staged_model.lume_model_instances) == 2
        assert isinstance(staged_model.lume_model_instances[0], InjectorSurrogate)
        assert staged_model.lume_model_instances[1] is not None

    def test_output_beam_variable_type_validation(self, injector_model):
        """Test that output_beam must be ParticleGroupVariable."""
        # Get output_beam from injector surrogate and check type
        assert isinstance(injector_model.final_particles, ParticleGroup)

    def test_input_beam_variable_present(self, cu_hxr_bmad_model):
        """Test that downstream models have input_beam variable."""
        # cu_hxr model should have input_beam
        assert isinstance(cu_hxr_bmad_model.initial_particles, ParticleGroup)


class TestStagedModelVariables:
    """Test variable access and management across staged models."""

    def test_supported_variables_aggregation(self, staged_model):
        """Test that supported_variables combines all model variables."""
        supported_vars = staged_model.supported_variables

        # Check for variables from injector
        injector_vars = staged_model.lume_model_instances[0].supported_variables
        for var_name in injector_vars:
            assert var_name in supported_vars

        # Check for variables from cu_hxr
        cu_hxr_vars = staged_model.lume_model_instances[1].supported_variables
        for var_name in cu_hxr_vars:
            assert var_name in supported_vars

    def test_get_injector_variables(self, staged_model):
        """Test getting variables from the injector stage."""
        injector = staged_model.lume_model_instances[0]

        # Get injector-specific variable
        injector_vars = list(injector.supported_variables.keys())[:1]
        if injector_vars:
            result = staged_model.get(injector_vars)
            assert isinstance(result, dict)
            assert injector_vars[0] in result

    def test_get_cu_hxr_variables(self, staged_model):
        """Test getting variables from the cu_hxr stage."""
        # Get cu_hxr-specific observable
        cu_hxr_vars = ["a.beta", "b.beta"]
        result = staged_model.get(cu_hxr_vars)
        assert "a.beta" in result
        assert "b.beta" in result

    def test_staged_model_edge_case(self):
        model = get_cu_hxr_staged_model(end_element="TD11")
        model.set({"QUAD:IN20:525:BCTRL": 10})
        b = model.get("x.beta")
        assert b is not None


class TestStagedModelStaging:
    """Test beam staging and propagation through models."""

    def test_cu_hxr_staged_model_output(self, staged_model):
        """Test staged model beam output after setting control variables."""
        model = staged_model

        SCAN_QUAD_PV = "QUAD:IN20:525:BCTRL"
        model.set({SCAN_QUAD_PV: float(-10)})
        b = model.get("OTR4_beam")

        assert b["norm_emit_y"] is not None

    def test_get_cu_hxr_observable(self, staged_model):
        """Test retrieving cu_hxr observable variables."""
        # Try to get an observable variable
        result = staged_model.get(["a.beta"])
        assert "a.beta" in result
        assert result["a.beta"] is not None

    def test_beam_propagation_output(self, staged_model):
        """Test that beams propagate through stages with tracking enabled."""
        model = staged_model

        # Set a quad and verify output beam exists
        model.set({"QUAD:IN20:631:BCTRL": -5.0})

        # Get tracked screen beam at OTR4
        result = model.get("OTR4_beam")
        assert result is not None
        # result is a ParticleGroup object from openPMD
        assert "ParticleGroup" in str(type(result))

    def test_multiple_observable_get(self, staged_model):
        """Test getting multiple observables from different stages."""
        # Get multiple lattice parameters
        result = staged_model.get(["a.beta", "b.beta"])
        assert "a.beta" in result
        assert "b.beta" in result

    @pytest.mark.skipif(
        not os.environ.get("FACET2_LATTICE"),
        reason="requires FACET2_LATTICE",
    )
    def test_facet_model(self):
        staged_model = get_facet_staged_model(end_element="PR10711")
        staged_model.get(list(staged_model.supported_variables.keys()))
