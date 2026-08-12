from typing import Any

import numpy as np

from lume.model import LUMEModel
from lume.variables import NDVariable, ScalarVariable
from zfel import sase1d

C_LIGHT = 299_792_458.0

from virtual_accelerator.zfel.undulator_mapping import (
    MAGNETIC_LENGTH_M,
    N_ACTIVE_SEGMENTS,
    build_hxr_mapping,
)


class ZFELModel(LUMEModel):
    """
    LUMEModel wrapper around zfel using real-machine-like
    HXR Kact and DSKact controls.

    Flow
    ----
    Kact / DSKact
        -> realistic HXR machine mapping
        -> effective zfel magnetic coordinate
        -> zfel simulation
        -> cached FEL diagnostics
    """

    ZFEL_Z_STEPS = 320
    INITIAL_K = 3.5

    def __init__(self):
        # ----------------------------------------------------------
        # Baseline zfel beam inputs
        #
        # These are still the generic zfel test parameters.
        # Only the undulator representation is being upgraded in
        # Phase 4D.
        # ----------------------------------------------------------

        self._sase_input = {
            "npart": 512,
            "s_steps": 200,
            "z_steps": self.ZFEL_Z_STEPS,
            "energy": 4313.34e6,
            "eSpread": 0.0,
            "emitN": 1.2e-6,
            "currentMax": 3400,
            "beta": 26,
            "unduPeriod": 0.03,
            "unduK": self.INITIAL_K,
            "unduL": MAGNETIC_LENGTH_M,
            "radWavelength": None,
            "random_seed": 31,
            "particle_position": None,
            "hist_rule": "square-root",
            "iopt": "sase",
            "P0": 0,
        }

        # ----------------------------------------------------------
        # Initial virtual-machine controls
        #
        # Constant K is used only to validate the new Kact/DSKact
        # control path.
        # ----------------------------------------------------------

        initial_kact = np.full(
            N_ACTIVE_SEGMENTS,
            self.INITIAL_K,
            dtype=float,
        )

        initial_dskact = np.full(
            N_ACTIVE_SEGMENTS,
            self.INITIAL_K,
            dtype=float,
        )

        self._initial_state = {
            "Kact": initial_kact,
            "DSKact": initial_dskact,
            "unduK": np.full(
                self.ZFEL_Z_STEPS,
                self.INITIAL_K,
                dtype=float,
            ),
            "power_max": 0.0,
            "exit_power": 0.0,
            "pulse_energy": 0.0,
        }

        self._state = self._copy_initial_state()

        # ----------------------------------------------------------
        # LUMEModel variable definitions
        # ----------------------------------------------------------

        self._variables = {
            # Writable virtual-machine controls
            "Kact": NDVariable(
                name="Kact",
                shape=(N_ACTIVE_SEGMENTS,),
                default_value=initial_kact.copy(),
                unit="dimensionless",
                read_only=False,
            ),
            "DSKact": NDVariable(
                name="DSKact",
                shape=(N_ACTIVE_SEGMENTS,),
                default_value=initial_dskact.copy(),
                unit="dimensionless",
                read_only=False,
            ),

            # Read-only mapped zfel K profile
            "unduK": NDVariable(
                name="unduK",
                shape=(self.ZFEL_Z_STEPS,),
                default_value=np.full(
                    self.ZFEL_Z_STEPS,
                    self.INITIAL_K,
                    dtype=float,
                ),
                unit="dimensionless",
                read_only=True,
            ),

            # Read-only FEL diagnostics
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
        }

        # Store the latest complete HXR mapping for debugging,
        # plotting, and later Badger integration.
        self._mapping = None

        # Run the initial constant-K case.
        self._run_simulation()

    @property
    def supported_variables(self):
        return self._variables

    def _copy_initial_state(self) -> dict[str, Any]:
        """
        Return an independent copy of the initial machine state.
        """

        return {
            "Kact": self._initial_state["Kact"].copy(),
            "DSKact": self._initial_state["DSKact"].copy(),
            "unduK": self._initial_state["unduK"].copy(),
            "power_max": float(self._initial_state["power_max"]),
            "exit_power": float(self._initial_state["exit_power"]),
            "pulse_energy": float(self._initial_state["pulse_energy"]),
        }

    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Return cached state only.

        Calling get() does not rerun zfel.
        """

        out = {}

        for name in names:
            value = self._state[name]

            if isinstance(value, np.ndarray):
                out[name] = value.copy()
            else:
                out[name] = value

        return out

    def _set(self, values: dict[str, Any]) -> None:
        """
        Update Kact and/or DSKact, then run zfel once.
        """

        if "Kact" in values:
            self._state["Kact"] = np.asarray(
                values["Kact"],
                dtype=float,
            ).copy()

        if "DSKact" in values:
            self._state["DSKact"] = np.asarray(
                values["DSKact"],
                dtype=float,
            ).copy()

        self._run_simulation()

    def _run_simulation(self) -> None:
        """
        Map the virtual HXR machine state into zfel and run once.
        """

        mapping = build_hxr_mapping(
            self._state["Kact"],
            self._state["DSKact"],
            z_steps=self.ZFEL_Z_STEPS,
        )

        self._mapping = mapping

        k_zfel = np.asarray(
            mapping["k_zfel"],
            dtype=float,
        )

        self._state["unduK"] = k_zfel.copy()

        sase_input = self._sase_input.copy()
        sase_input["unduK"] = k_zfel

        output = sase1d.sase(sase_input)

        power_z = np.asarray(
            output["power_z"],
            dtype=float,
        )

        self._state["power_max"] = float(
            np.max(power_z)
        )

        self._state["exit_power"] = float(
            power_z[-1]
        )
        power_s = np.asarray(
            output["power_s"],
            dtype=float,
        )

        s_m = np.asarray(
            output["s"],
            dtype=float,
        )

        if s_m.size < 2:
            raise ValueError(
                "zfel must return at least two longitudinal s points."
            )

        ds_m = float(
            np.mean(np.diff(s_m))
        )

        # Each power_s sample represents one longitudinal slice.
        # Convert ds [m] to dt [s] using dt = ds/c.
        pulse_energy_j = float(
            np.sum(power_s[-1, :])
            * ds_m
            / C_LIGHT
        )

        self._state["pulse_energy"] = pulse_energy_j

    def reset(self) -> None:
        """
        Restore the initial 32-segment virtual-machine state.
        """

        self._state = self._copy_initial_state()
        self._run_simulation()