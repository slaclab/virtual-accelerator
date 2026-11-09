from typing import Any

import time

import numpy as np

from lume.model import LUMEModel
from lume.variables import ScalarVariable

from virtual_accelerator.zfel.undulator_mapping import HXR_CELLS
from virtual_accelerator.zfel.model import ZFELBackend


class ZFELPVModel(LUMEModel):
    """
    Scalar EPICS-facing wrapper around ZFELModel.

    EPICS-facing controls:
        KAct_14, DSKAct_14, ..., KAct_47, DSKAct_47

    Internal physics model:
        KAct[32], DSKact[32] -> zfel
    """

    def __init__(self,
        simulate_motion=False,
        motion_t_avg=30.0,
        motion_jitter=0.2,
    ):
        if motion_t_avg < 0:
            raise ValueError("motion_t_avg must be nonnegative")
        if not 0.0 <= motion_jitter <= 1.0:
            raise ValueError("motion_jitter must be between 0 and 1")

        self.simulate_motion = simulate_motion
        self.motion_t_avg = float(motion_t_avg)
        self.motion_jitter = float(motion_jitter)

        self._rng = np.random.default_rng()
        self._pending_controls = None
        self._motion_deadline = None

        self._backend = ZFELBackend()

        backend_state = self._backend.get(
            [
                "KAct",
                "DSKAct",
                "power_max",
                "exit_power",
                "pulse_energy",
            ]
        )

        kact = np.asarray(
            backend_state["KAct"],
            dtype=float,
        )

        dskact = np.asarray(
            backend_state["DSKAct"],
            dtype=float,
        )

        self._state: dict[str, Any] = {}
        self._pv_aliases = {}
        self._variables = {}
        self._state["model_eval_id"] = 0.0
        self._state["motion_busy"] = 0.0
        self._state["refresh"] = 0.0

        # ------------------------------------------------------
        # Scalar, real-machine-like undulator controls
        # ------------------------------------------------------

        for index, cell in enumerate(HXR_CELLS):
            kact_name = f"KAct_{cell}"
            dskact_name = f"DSKAct_{cell}"

            self._state[kact_name] = float(kact[index])
            self._state[dskact_name] = float(dskact[index])

            kact_pv = f"USEG:UNDH:{cell}50:KAct"
            dskact_pv = f"USEG:UNDH:{cell}50:DSKAct"

            self._pv_aliases[kact_pv] = kact_name
            self._pv_aliases[dskact_pv] = dskact_name

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

            self._variables[kact_pv] = ScalarVariable(
                name=kact_pv,
                default_value=float(kact[index]),
                value_range=(0.0, 5.0),
                unit="dimensionless",
                read_only=False,
            )

            self._variables[dskact_pv] = ScalarVariable(
                name=dskact_pv,
                default_value=float(dskact[index]),
                value_range=(0.0, 5.0),
                unit="dimensionless",
                read_only=False,
            )

        # ------------------------------------------------------
        # Read-only FEL diagnostics
        # ------------------------------------------------------

        self._variables.update(
            {
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
                "model_eval_id": ScalarVariable(
                    name="model_eval_id",
                    default_value=0.0,
                    unit="",
                    read_only=True,
                ),
            }
        )

        self._sync_from_backend(include_controls=True)

        self._pv_aliases.update(
            {
                "GDET:FEE1:361:ENRC": "pulse_intensity_mean",
                "GDET:FEE1:361:ENRCHSTCUHBR": "pulse_intensity_p80",
                "ZFEL:POWER_MAX": "power_max",
                "ZFEL:EXIT_POWER": "exit_power",
                "ZFEL:PULSE_ENERGY": "pulse_energy",
                "ZFEL:PULSE_INTENSITY_STD_REL": "pulse_intensity_std_relative",
                "ZFEL:MODEL_EVAL_ID": "model_eval_id",
            }
        )

        self._variables.update(
            {
                "GDET:FEE1:361:ENRC": ScalarVariable(
                    name="GDET:FEE1:361:ENRC",
                    default_value=0.0,
                    unit="J",
                    read_only=True,
                ),
                "GDET:FEE1:361:ENRCHSTCUHBR": ScalarVariable(
                    name="GDET:FEE1:361:ENRCHSTCUHBR",
                    default_value=0.0,
                    unit="J",
                    read_only=True,
                ),
                "ZFEL:POWER_MAX": ScalarVariable(
                    name="ZFEL:POWER_MAX",
                    default_value=0.0,
                    unit="W",
                    read_only=True,
                ),
                "ZFEL:EXIT_POWER": ScalarVariable(
                    name="ZFEL:EXIT_POWER",
                    default_value=0.0,
                    unit="W",
                    read_only=True,
                ),
                "ZFEL:PULSE_ENERGY": ScalarVariable(
                    name="ZFEL:PULSE_ENERGY",
                    default_value=0.0,
                    unit="J",
                    read_only=True,
                ),
                "ZFEL:PULSE_INTENSITY_STD_REL": ScalarVariable(
                    name="ZFEL:PULSE_INTENSITY_STD_REL",
                    default_value=0.0,
                    unit="dimensionless",
                    read_only=True,
                ),
                "ZFEL:MODEL_EVAL_ID": ScalarVariable(
                    name="ZFEL:MODEL_EVAL_ID",
                    default_value=0.0,
                    unit="",
                    read_only=True,
                ),
            }
        )

        self._pv_aliases["ZFEL:MOTION_BUSY"] = "motion_busy"

        self._variables["ZFEL:MOTION_BUSY"] = ScalarVariable(
            name="ZFEL:MOTION_BUSY",
            default_value=0.0,
            unit="",
            read_only=True,
        )

        self._pv_aliases["ZFEL:REFRESH"] = "refresh"

        self._variables["ZFEL:REFRESH"] = ScalarVariable(
            name="ZFEL:REFRESH",
            default_value=0.0,
            unit="",
            read_only=False,
        )

    @property
    def supported_variables(self):
        return self._variables

    def get(self, names):
        """
        Compatibility wrapper for Lume-PVA.

        lume-pva may pass supported_variables directly to Model.get(),
        while lume expects a list of variable-name strings.
        """
        if isinstance(names, dict):
            names = list(names)

        return super().get(names)

    def _get(self, names):
        self._complete_motion_if_ready()

        return {
            name: self._state[self._pv_aliases.get(name, name)]
            for name in names
        }


    def _complete_motion_if_ready(self):
        """Finish simulated motion when its deadline has passed."""
        if self._motion_deadline is None:
            return

        if time.monotonic() < self._motion_deadline:
            return

        self._backend.set(self._pending_controls)
        self._sync_from_backend(include_controls=True)

        self._state["model_eval_id"] += 1.0
        self._state["motion_busy"] = 0.0

        self._pending_controls = None
        self._motion_deadline = None

        print("VA: Simulated motion complete", flush=True)

    def _set(self, values: dict[str, Any]) -> None:
        """Stage K changes and simulate motion before ZFEL evaluation."""
        self._complete_motion_if_ready()

        # Remove refresh commands before processing K changes.
        values = {
            name: value
            for name, value in values.items()
            if self._pv_aliases.get(name, name) != "refresh"
        }

        if not values:
            return

    # Keep the existing KAct/DSKAct processing below unchanged.
        if self._pending_controls is not None:
            kact = self._pending_controls["KAct"].copy()
            dskact = self._pending_controls["DSKAct"].copy()
        else:
            kact = np.asarray(
                [self._state[f"KAct_{cell}"] for cell in HXR_CELLS],
                dtype=float,
            )
            dskact = np.asarray(
                [self._state[f"DSKAct_{cell}"] for cell in HXR_CELLS],
                dtype=float,
            )

        cell_indices = {
            cell: index for index, cell in enumerate(HXR_CELLS)
        }

        for name, value in values.items():
            state_name = self._pv_aliases.get(name, name)

            if state_name.startswith("KAct_"):
                cell = int(state_name.split("_")[1])
                kact[cell_indices[cell]] = float(value)

            elif state_name.startswith("DSKAct_"):
                cell = int(state_name.split("_")[1])
                dskact[cell_indices[cell]] = float(value)

            else:
                raise KeyError(f"Unsupported control: {name}")

        controls = {
            "KAct": kact,
            "DSKAct": dskact,
        }

        if self.simulate_motion:
            self._pending_controls = controls

            if self._motion_deadline is None:
                delay = self.motion_t_avg * self._rng.uniform(
                    1.0 - self.motion_jitter,
                    1.0 + self.motion_jitter,
                )

                self._motion_deadline = time.monotonic() + delay
                self._state["motion_busy"] = 1.0

                print(
                    f"VA: Simulated motion time = {delay:.1f} s",
                    flush=True,
                )

            return

        self._backend.set(controls)
        self._sync_from_backend(include_controls=True)
        self._state["model_eval_id"] += 1.0

    def _sync_from_backend(
        self,
        *,
        include_controls: bool,
    ) -> None:
        backend_state = self._backend.get(
            [
                "KAct",
                "DSKAct",
                "power_max",
                "exit_power",
                "pulse_energy",
            ]
        )

        if include_controls:
            kact = np.asarray(
                backend_state["KAct"],
                dtype=float,
            )

            dskact = np.asarray(
                backend_state["DSKAct"],
                dtype=float,
            )

            for index, cell in enumerate(HXR_CELLS):
                self._state[f"KAct_{cell}"] = float(kact[index])

                self._state[f"DSKAct_{cell}"] = float(dskact[index])

        pulse_energy = float(backend_state["pulse_energy"])

        self._state["power_max"] = float(backend_state["power_max"])

        self._state["exit_power"] = float(backend_state["exit_power"])

        self._state["pulse_energy"] = pulse_energy

        # Fixed-seed deterministic model for now.
        self._state["pulse_intensity_mean"] = pulse_energy
        self._state["pulse_intensity_p80"] = pulse_energy
        self._state["pulse_intensity_std_relative"] = 0.0

    def reset(self) -> None:
        self._pending_controls = None
        self._motion_deadline = None
        self._state["motion_busy"] = 0.0
        self._backend.reset()

        self._sync_from_backend(include_controls=True)


def get_cu_hxr_zfel_model() -> ZFELPVModel:
    """
    Construct the CU HXR ZFEL virtual-accelerator model.
    """
    return ZFELPVModel(
        simulate_motion=True,
        motion_t_avg=10.0,
        motion_jitter=0.2,
    )
