import numpy as np
import pytest
from lume.variables import NDVariable, ScalarVariable, StrVariable

from virtual_accelerator.models.static_model import load_pv_values, StaticVariableModel


def test_supported_variables_are_inferred_from_initial_values():
    model = StaticVariableModel(
        {
            "scalar_var": 1.5,
            "bool_var": True,
            "str_var": "abc",
            "nd_var": [1.0, 2.0, 3.0],
        }
    )

    supported = model.supported_variables

    assert isinstance(supported["scalar_var"], ScalarVariable)
    assert isinstance(supported["bool_var"], ScalarVariable)
    assert isinstance(supported["str_var"], StrVariable)
    assert isinstance(supported["nd_var"], NDVariable)
    assert tuple(supported["nd_var"].shape) == (3,)


def test_get_and_set_round_trip_values():
    model = StaticVariableModel(
        {
            "x": 1.0,
            "name": "a",
            "arr": np.array([[1, 2], [3, 4]]),
        }
    )

    assert model.get("x") == 1.0
    assert model.get(["x", "name"]) == {"x": 1.0, "name": "a"}

    model.set({"x": 2.5, "name": "b"})

    assert model.get("x") == 2.5
    assert model.get("name") == "b"


def test_partial_set_preserves_other_values():
    model = StaticVariableModel({"x": 1.0, "y": 2.0})

    model.set({"x": 10.0})

    assert model.get("x") == 10.0
    assert model.get("y") == 2.0


def test_reset_restores_initial_values():
    model = StaticVariableModel({"x": 1.0, "name": "start"})

    model.set({"x": 3.0, "name": "updated"})
    model.reset()

    assert model.get("x") == 1.0
    assert model.get("name") == "start"


def test_unsupported_initial_value_type_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported value type"):
        StaticVariableModel({"bad": object()})


def test_load_pv_values_from_csv_file(tmp_path):
    file_path = tmp_path / "pv_values.csv"
    file_path.write_text(
        'PV:ONE,1.25\nPV:TWO,true\nPV:THREE,"[1, 2, 3]"\n', encoding="utf-8"
    )

    values = load_pv_values(str(file_path))

    assert values["PV:ONE"] == 1.25
    assert values["PV:TWO"] is True
    assert values["PV:THREE"] == [1, 2, 3]


def test_load_pv_values_rejects_invalid_csv_row(tmp_path):
    file_path = tmp_path / "pv_values.csv"
    file_path.write_text("PV:ONE,2,extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid CSV row"):
        load_pv_values(str(file_path))


def test_load_pv_values_rejects_duplicate_pv_name(tmp_path):
    file_path = tmp_path / "pv_values.csv"
    file_path.write_text("PV:ONE,1\nPV:ONE,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate PV name"):
        load_pv_values(str(file_path))
