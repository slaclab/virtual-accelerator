import math

import pytest

from virtual_accelerator.tests.dependency_profiles import HAS_CHEETAH_DEPS

pytestmark = [
    pytest.mark.requires_cheetah,
]

if HAS_CHEETAH_DEPS:
    import torch
    from cheetah.accelerator import Drift, Quadrupole, Segment
    from cheetah.particles import ParticleBeam
    from lume_cheetah.simulator import CheetahSimulator

    from virtual_accelerator.cheetah.actions import (
        QuadrupoleBACTVariable,
        QuadrupoleBCTRLVariable,
    )
    from virtual_accelerator.cheetah.variables import get_variables_from_segment
else:
    pytest.skip("requires cheetah optional dependencies", allow_module_level=True)


class TestCheetahActions:
    @pytest.fixture
    def simulator(self):
        segment = Segment(
            [
                Quadrupole(
                    name="Q1",
                    length=torch.tensor(0.5),
                    k1=torch.tensor(1.0),
                ),
                Drift(name="D1", length=torch.tensor(1.0)),
            ]
        )
        beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            num_particles=256,
            energy=torch.tensor(1e6),
        )
        return CheetahSimulator(segment=segment, initial_beam_distribution=beam)

    def test_quadrupole_bctrl_roundtrip(self, simulator):
        variable = QuadrupoleBCTRLVariable(
            name="QUAD:IN20:511:BCTRL",
            element_name="Q1",
            pv_attribute="BCTRL",
        )

        initial_value = float(variable._get(simulator))
        expected_initial = 1.0 * 0.5 * 33.356 * 1e6 / 1e9
        assert math.isclose(initial_value, expected_initial, rel_tol=0.0, abs_tol=1e-8)

        variable._set(simulator, torch.tensor(0.05))
        roundtrip = float(variable._get(simulator))

        assert math.isclose(roundtrip, 0.05, rel_tol=0.0, abs_tol=1e-8)

    def test_quadrupole_bact_is_read_only(self, simulator):
        variable = QuadrupoleBACTVariable(
            name="QUAD:IN20:511:BACT",
            element_name="Q1",
            pv_attribute="BACT",
        )

        with pytest.raises(RuntimeError):
            variable._set(simulator, torch.tensor(0.1))

    def test_get_variables_from_segment_instantiates_actions(self):
        segment = Segment(
            [
                Quadrupole(
                    name="Q1",
                    length=torch.tensor(0.5),
                    k1=torch.tensor(1.0),
                ),
                Drift(name="D1", length=torch.tensor(1.0)),
            ]
        )

        variables = get_variables_from_segment(
            segment,
            device_mapping={"Q1": "QUAD:IN20:511"},
            element_attr_mapping={
                "Quadrupole": {
                    "BCTRL": "QuadrupoleBCTRLVariable",
                    "BACT": "QuadrupoleBACTVariable",
                }
            },
        )

        assert "QUAD:IN20:511:BCTRL" in variables
        assert "QUAD:IN20:511:BACT" in variables
        assert isinstance(variables["QUAD:IN20:511:BCTRL"], QuadrupoleBCTRLVariable)
        assert isinstance(variables["QUAD:IN20:511:BACT"], QuadrupoleBACTVariable)

    def test_split_quadrupole_get_and_set(self):
        segment = Segment(
            [
                Quadrupole(
                    name="Q1#1",
                    length=torch.tensor(0.2),
                    k1=torch.tensor(1.0),
                ),
                Quadrupole(
                    name="Q1#2",
                    length=torch.tensor(0.3),
                    k1=torch.tensor(1.0),
                ),
                Drift(name="D1", length=torch.tensor(1.0)),
            ]
        )
        beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            num_particles=256,
            energy=torch.tensor(1e6),
        )
        simulator = CheetahSimulator(segment=segment, initial_beam_distribution=beam)

        variable = QuadrupoleBCTRLVariable(
            name="QUAD:IN20:511:BCTRL",
            element_name="Q1",
            pv_attribute="BCTRL",
        )

        initial_value = float(variable._get(simulator))
        expected_initial = 1.0 * 0.5 * 33.356 * 1e6 / 1e9
        assert math.isclose(initial_value, expected_initial, rel_tol=0.0, abs_tol=1e-8)

        variable._set(simulator, torch.tensor(0.06))

        q1_k1_values = [element.k1 for element in simulator.segment.elements[:2]]
        assert torch.equal(q1_k1_values[0], q1_k1_values[1])
        assert math.isclose(float(variable._get(simulator)), 0.06, rel_tol=0.0, abs_tol=1e-8)

    def test_split_quadrupole_variables_are_deduped_by_control_name(self):
        segment = Segment(
            [
                Quadrupole(
                    name="Q1#1",
                    length=torch.tensor(0.2),
                    k1=torch.tensor(1.0),
                ),
                Quadrupole(
                    name="Q1#2",
                    length=torch.tensor(0.3),
                    k1=torch.tensor(1.0),
                ),
            ]
        )

        variables = get_variables_from_segment(
            segment,
            device_mapping={"Q1": "QUAD:IN20:511"},
            element_attr_mapping={
                "Quadrupole": {
                    "BCTRL": "QuadrupoleBCTRLVariable",
                }
            },
        )

        assert list(variables.keys()) == ["QUAD:IN20:511:BCTRL"]
