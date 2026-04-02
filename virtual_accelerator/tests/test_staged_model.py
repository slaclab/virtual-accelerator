import os
from pathlib import Path
import pytest
from lume.variables.particle_group import ParticleGroupVariable

from virtual_accelerator.models.staged_model import (
    StagedModel,
    get_cu_hxr_staged_model,
)
from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model
from virtual_accelerator.surrogates.injector_surrogate import InjectorSurrogate

TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "../bmad", "test_beam")


# Fixtures for model initialization
@pytest.fixture
def injector_model():
    """Fixture providing an InjectorSurrogate instance."""
    return InjectorSurrogate()


@pytest.fixture
def cu_hxr_bmad_model():
    """Fixture providing a cu_hxr BMAD model with custom beam path."""
    return get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)


@pytest.fixture
def staged_model(injector_model, cu_hxr_bmad_model):
    """Fixture providing a basic StagedModel combining injector and cu_hxr."""
    return StagedModel([injector_model, cu_hxr_bmad_model])


@pytest.fixture
def staged_model_with_tracking():
    """Fixture providing a StagedModel with beam tracking enabled."""
    return get_cu_hxr_staged_model(track_beam=True, end_element="OTR4")


class TestStagedModelValidation:
    """Test StagedModel validation and initialization."""

    def test_staged_model_initialization(self, staged_model):
        """Test valid StagedModel initialization."""
        assert len(staged_model.lume_model_instances) == 2
        assert isinstance(staged_model.lume_model_instances[0], InjectorSurrogate)
        assert staged_model.lume_model_instances[1] is not None

    def test_output_beam_variable_type_validation(self, injector_model):
        """Test that output_beam must be ParticleGroupVariable."""
        # Get output_beam from injector and verify it's ParticleGroupVariable
        output_beam = injector_model.supported_variables.get("output_beam")
        assert isinstance(output_beam, ParticleGroupVariable)

    def test_input_beam_variable_present(self, cu_hxr_bmad_model):
        """Test that downstream models have input_beam variable."""
        # cu_hxr model should have input_beam
        assert "input_beam" in cu_hxr_bmad_model.supported_variables
        assert isinstance(
            cu_hxr_bmad_model.supported_variables["input_beam"], ParticleGroupVariable
        )


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


class TestStagedModelStaging:
    """Test beam staging and propagation through models."""

    def test_cu_hxr_staged_model_output(self, staged_model_with_tracking):
        """Test staged model beam output after setting control variables."""
        model = staged_model_with_tracking

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

    def test_beam_propagation_output(self, staged_model_with_tracking):
        """Test that beams propagate through stages with tracking enabled."""
        model = staged_model_with_tracking

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


class TestStagedModelFactory:
    """Test the factory function get_cu_hxr_staged_model."""

    def test_get_cu_hxr_staged_model_basic(self):
        """Test factory function creates valid StagedModel."""
        model = get_cu_hxr_staged_model(custom_beam_path=TEST_BEAM_PATH)

        assert isinstance(model, StagedModel)
        assert len(model.lume_model_instances) == 2

    def test_get_cu_hxr_staged_model_with_track_beam(self):
        """Test factory function with track_beam=True."""
        model = get_cu_hxr_staged_model(track_beam=True, end_element="OTR4")

        assert isinstance(model, StagedModel)
        cu_hxr = model.lume_model_instances[1]
        # Verify tracking is enabled
        assert "OTR4_beam" in cu_hxr.supported_variables

    def test_get_cu_hxr_staged_model_with_end_element(self):
        """Test factory function with custom end_element."""
        model = get_cu_hxr_staged_model(end_element="OTR2")

        assert isinstance(model, StagedModel)
        cu_hxr = model.lume_model_instances[1]
        # Model should be created successfully with custom end_element
        assert len(cu_hxr.supported_variables) > 0
