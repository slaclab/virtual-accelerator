from typing import Any

import numpy as np

from lume.model import LUMEModel
from lume.variables import ScalarVariable

from virtual_accelerator.zfel.undulator_mapping import HXR_CELLS
from virtual_accelerator.zfel.model import ZFELModel


class ZFELPVModel(LUMEModel):
    """
    Scalar EPICS-facing wrapper around ZFELModel.

    EPICS-facing controls:
        KAct_14, DSKAct_14, ..., KAct_47, DSKAct_47

    Internal physics model:
        Kact[32], DSKact[32] -> zfel
    """

    def __init__(self):
        self._backend = ZFELModel()

        backend_state = self._backend.get([
            "Kact",
            "DSKact",
            "power_max",
            "exit_power",
            "pulse_energy",
        ])

        kact = np.asarray(
            backend_state["Kact"],
            dtype=float,
        )

        dskact = np.asarray(
            backend_state["DSKact"],
            dtype=float,
        )

        self._state: dict[str, Any] = {}
        self._variables = {}

        # ------------------------------------------------------
        # Scalar, real-machine-like undulator controls
        # ------------------------------------------------------

        for index, cell in enumerate(HXR_CELLS):
            kact_name = f"KAct_{cell}"
            dskact_name = f"DSKAct_{cell}"

            self._state[kact_name] = float(kact[index])
            self._state[dskact_name] = float(dskact[index])

            self._variables[kact_name] = ScalarVariable(
                name=kact_name,
                default_value=float(kact[index]),
                value_range=(0.0, 5.0),
                unit="dimensionless",
                read_only=False,
            )

            self._variables[dskact_name] = ScalarVariable(
                name=dskact_name,
                default_value=float(dskact[index]),
                value_range=(0.0, 5.0),
                unit="dimensionless",
                read_only=False,
            )

        # ------------------------------------------------------
        # Read-only FEL diagnostics
        # ------------------------------------------------------

        self._variables.update({
            "power_max": ScalarVariable(
                name="power_max",
                default_value=0.0,
                unit="W",
                read_only=True,
            ),
            "exit_power": ScalarVariable(
                name="exit_power",
                default_value=0.0,
                unit="W",
                read_only=True,
            ),
            "pulse_energy": ScalarVariable(
                name="pulse_energy",
                default_value=0.0,
                unit="J",
                read_only=True,
            ),
            "pulse_intensity_mean": ScalarVariable(
                name="pulse_intensity_mean",
                default_value=0.0,
                unit="J",
                read_only=True,
            ),
            "pulse_intensity_p80": ScalarVariable(
                name="pulse_intensity_p80",
                default_value=0.0,
                unit="J",
                read_only=True,
            ),
            "pulse_intensity_std_relative": ScalarVariable(
                name="pulse_intensity_std_relative",
                default_value=0.0,
                unit="dimensionless",
                read_only=True,
            ),
        })

        self._sync_from_backend(include_controls=True)

    @property
    def supported_variables(self):
        return self._variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {
            name: self._state[name]
            for name in names
        }

    def _set(self, values: dict[str, Any]) -> None:
        """
        Update any scalar KAct/DSKAct values, then run zfel once.

        Lume-PVA batches multiple PV writes before calling this,
        according to Runner update_rate.
        """

        if not values:
            self._sync_from_backend(
                include_controls=True
            )
            return

        for name, value in values.items():
            self._state[name] = float(value)

        kact = np.asarray(
            [
                self._state[f"KAct_{cell}"]
                for cell in HXR_CELLS
            ],
            dtype=float,
        )

        dskact = np.asarray(
            [
                self._state[f"DSKAct_{cell}"]
                for cell in HXR_CELLS
            ],
            dtype=float,
        )

        self._backend.set({
            "Kact": kact,
            "DSKact": dskact,
        })

        self._sync_from_backend(
            include_controls=True
        )

    def _sync_from_backend(
        self,
        *,
        include_controls: bool,
    ) -> None:
        backend_state = self._backend.get([
            "Kact",
            "DSKact",
            "power_max",
            "exit_power",
            "pulse_energy",
        ])

        if include_controls:
            kact = np.asarray(
                backend_state["Kact"],
                dtype=float,
            )

            dskact = np.asarray(
                backend_state["DSKact"],
                dtype=float,
            )

            for index, cell in enumerate(HXR_CELLS):
                self._state[f"KAct_{cell}"] = float(
                    kact[index]
                )

                self._state[f"DSKAct_{cell}"] = float(
                    dskact[index]
                )

        pulse_energy = float(
            backend_state["pulse_energy"]
        )

        self._state["power_max"] = float(
            backend_state["power_max"]
        )

        self._state["exit_power"] = float(
            backend_state["exit_power"]
        )

        self._state["pulse_energy"] = pulse_energy

        # Fixed-seed deterministic model for now.
        self._state["pulse_intensity_mean"] = (
            pulse_energy
        )
        self._state["pulse_intensity_p80"] = (
            pulse_energy
        )
        self._state[
            "pulse_intensity_std_relative"
        ] = 0.0

    def reset(self) -> None:
        self._backend.reset()

        self._sync_from_backend(
            include_controls=True
        )

def get_cu_hxr_zfel_model() -> ZFELPVModel:
    """
    Construct the CU HXR ZFEL virtual-accelerator model.
    """
    return ZFELPVModel()

def build_cu_hxr_zfel_runner_config(
    runner_cls,
    model,
    *,
    prefix: str = "VA:",
    protocols: tuple[str, ...] = ("ca", "pva"),
    update_rate: float = 0.5,
):
    """
    Build the lume-pva Runner configuration for the CU HXR ZFEL model.
    """

    config = runner_cls.generate_config(
        model=model,
        prefix="",
    )

    # PV names below already include the requested prefix.
    config["prefix"] = ""
    config["protocol"] = list(protocols)
    config["update_rate"] = update_rate

    def va_pv(name: str) -> str:
        return f"{prefix}{name}"

    for cell in HXR_CELLS:
        config["variables"][f"KAct_{cell}"]["pv"] = va_pv(
            f"USEG:UNDH:{cell}50:KAct"
        )

        config["variables"][f"DSKAct_{cell}"]["pv"] = va_pv(
            f"USEG:UNDH:{cell}50:DSKAct"
        )

    config["variables"]["power_max"]["pv"] = va_pv(
        "ZFEL:POWER_MAX"
    )

    config["variables"]["exit_power"]["pv"] = va_pv(
        "ZFEL:EXIT_POWER"
    )

    config["variables"]["pulse_energy"]["pv"] = va_pv(
        "ZFEL:PULSE_ENERGY"
    )

    config["variables"]["pulse_intensity_mean"]["pv"] = va_pv(
        "GDET:FEE1:361:ENRC"
    )

    config["variables"]["pulse_intensity_p80"]["pv"] = va_pv(
        "GDET:FEE1:361:ENRCHSTCUHBR"
    )

    config["variables"]["pulse_intensity_std_relative"]["pv"] = va_pv(
        "ZFEL:PULSE_INTENSITY_STD_REL"
    )

    return config

def get_cu_hxr_zfel_runner(
    runner_cls,
    *,
    prefix: str = "VA:",
    protocols: tuple[str, ...] = ("ca", "pva"),
    update_rate: float = 0.5,
):
    """
    Construct a lume-pva Runner for the CU HXR ZFEL virtual accelerator.
    """

    model = get_cu_hxr_zfel_model()

    config = build_cu_hxr_zfel_runner_config(
        runner_cls,
        model,
        prefix=prefix,
        protocols=protocols,
        update_rate=update_rate,
    )

    return runner_cls(
        model=model,
        config=config,
    )