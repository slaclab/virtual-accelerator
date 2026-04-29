import pytest
from numbers import Number

try:
    import torch
    from lume_cheetah import LUMECheetahModel
except ModuleNotFoundError as exc:
    pytest.skip(
        f"Skipping Cheetah model tests because dependency is missing: {exc.name}",
        allow_module_level=True,
    )


@pytest.fixture
def cheetah_model(model_name, model_factory):
    model = model_factory()
    assert isinstance(model, LUMECheetahModel)
    return model


class TestCheetahModelBasics:
    @pytest.mark.for_every_cheetah_model
    def test_has_variables(self, cheetah_model):
        assert cheetah_model.control_variables
        assert cheetah_model.observable_variables
        assert cheetah_model.supported_variables

        for name in cheetah_model.control_variables:
            assert name in cheetah_model.supported_variables

        for name in cheetah_model.observable_variables:
            assert name in cheetah_model.supported_variables

    @pytest.mark.for_every_cheetah_model
    def test_get_observable_variable(self, cheetah_model):
        observable_name = next(iter(cheetah_model.observable_variables))
        output = cheetah_model.get([observable_name])

        assert observable_name in output
        assert output[observable_name] is not None

        value = output[observable_name]
        if hasattr(value, "shape"):
            assert value.shape is not None

    @pytest.mark.for_every_cheetah_model
    def test_set_and_read_control_variable(self, cheetah_model):
        control_name = next(iter(cheetah_model.control_variables))
        current_value = cheetah_model.get([control_name])[control_name]

        if isinstance(current_value, torch.Tensor):
            current_value = float(current_value.item())
        elif isinstance(current_value, Number):
            current_value = float(current_value)
        else:
            pytest.skip(
                f"Skipping control variable {control_name} because its type is not numeric: {type(current_value)}"
            )

        new_value = (
            current_value + 1.0 if abs(current_value) < 1e6 else current_value * 0.9
        )
        cheetah_model.set({control_name: new_value})

        updated_value = cheetah_model.get([control_name])[control_name]
        if isinstance(updated_value, torch.Tensor):
            updated_value = float(updated_value.item())

        assert isinstance(updated_value, Number)
        assert abs(updated_value - new_value) < 1e-6

    @pytest.mark.for_every_cheetah_model
    def test_reset_restores_initial_state(self, cheetah_model):
        control_name = next(iter(cheetah_model.control_variables))
        original_value = cheetah_model.get([control_name])[control_name]

        if isinstance(original_value, torch.Tensor):
            original_value = float(original_value.item())
        elif isinstance(original_value, Number):
            original_value = float(original_value)
        else:
            pytest.skip(
                f"Skipping reset test for control variable {control_name} because its type is not numeric: {type(original_value)}"
            )

        cheetah_model.set({control_name: original_value + 1.0})
        cheetah_model.reset()

        reset_value = cheetah_model.get([control_name])[control_name]
        if isinstance(reset_value, torch.Tensor):
            reset_value = float(reset_value.item())

        assert abs(reset_value - original_value) < 1e-6
