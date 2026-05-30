import argparse
import csv
import copy
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import yaml
from lume.model import LUMEModel
from lume.variables import NDVariable, ScalarVariable, StrVariable, Variable

from virtual_accelerator.utils.optional_dependencies import import_optional_symbol


def _is_scalar_value(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _is_nd_value(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _infer_supported_variable(name: str, value: Any) -> Variable:
    if _is_scalar_value(value):
        return ScalarVariable(name=name, unit="", read_only=False)

    if isinstance(value, str):
        return StrVariable(name=name, unit="", read_only=False)

    if _is_nd_value(value):
        shape = np.asarray(value).shape
        return NDVariable(name=name, shape=shape, unit="", read_only=False)

    raise ValueError(
        f"Unsupported value type for variable '{name}': {type(value).__name__}"
    )


class StaticVariableModel(LUMEModel):
    """A minimal LUMEModel that stores variable values in a local dictionary."""

    def __init__(self, initial_values: Mapping[str, Any]) -> None:
        super().__init__()
        self._initial_values = copy.deepcopy(dict(initial_values))
        self._cache = copy.deepcopy(self._initial_values)
        self._supported_variables = {
            name: _infer_supported_variable(name, value)
            for name, value in self._initial_values.items()
        }

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        self._cache.update(dict(values))

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return dict(self._supported_variables)

    def reset(self) -> None:
        self._cache = copy.deepcopy(self._initial_values)


def load_pv_values(file_path: str) -> dict[str, Any]:
    """Load PV names and values from a CSV file with two columns: PV,VALUE."""
    result: dict[str, Any] = {}
    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_no, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) != 2:
                raise ValueError(
                    f"Invalid CSV row {row_no} in '{file_path}'. "
                    "Expected exactly two columns: PV,VALUE."
                )

            name, raw_value = row
            name = name.strip()
            if not name:
                raise ValueError(f"Empty PV name on row {row_no} in '{file_path}'.")
            if name in result:
                raise ValueError(
                    f"Duplicate PV name '{name}' on row {row_no} in '{file_path}'."
                )

            result[name] = yaml.safe_load(raw_value.strip())

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a static PV/value mapping via lume-pva Runner."
    )
    parser.add_argument(
        "pv_values_file",
        help="Path to CSV file with two columns per row: PV,VALUE.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the CLI process (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    runner_cls = import_optional_symbol(
        "lume_pva.runner",
        "Runner",
        feature="static model PVA CLI",
        extra="pva",
    )

    initial_values = load_pv_values(args.pv_values_file)
    model = StaticVariableModel(initial_values)
    runner = runner_cls(model)
    runner.run()


if __name__ == "__main__":
    main()
