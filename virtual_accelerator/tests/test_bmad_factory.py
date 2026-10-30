from pathlib import Path

import numpy as np
import pytest

from virtual_accelerator.tests.dependency_profiles import HAS_BMAD_DEPS, HAS_LCLS_LATTICE

pytestmark = [pytest.mark.requires_bmad, pytest.mark.requires_lcls_lattice]

if not (HAS_BMAD_DEPS and HAS_LCLS_LATTICE):
    pytest.skip("requires Bmad and LCLS_LATTICE", allow_module_level=True)

from beamphysics import ParticleGroup
from lume_bmad.model import LUMEBmadModel
from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model

BEAM_DIR = Path(__file__).resolve().parents[1] / "bmad"


@pytest.mark.parametrize("source", ["custom", "default", "reference"])
def test_factory_sets_initial_particles(source, tmp_path, monkeypatch):
    calls = []
    original = LUMEBmadModel.initial_particles

    def set_particles(self, particles):
        assert self.tao.tao_global()["track_type"] == "beam"
        assert not self.tao.tao_global()["lattice_calc_on"]
        calls.append(particles.copy())
        original.fset(self, particles)

    monkeypatch.setattr(
        LUMEBmadModel, "initial_particles", property(original.fget, set_particles)
    )
    options = dict(start_element="OTR2", end_element="OTR4", track_beam=True)
    if source == "custom":
        path = tmp_path / "beam with spaces.h5"
        ParticleGroup(BEAM_DIR / "test_beam.h5").write(path)
        options["custom_beam_path"] = str(path)
    elif source == "reference":
        options["start_element"] = "DL02A2"

    if source == "reference":
        with pytest.warns(UserWarning, match="single reference particle"):
            model = get_cu_hxr_bmad_model(**options)
    else:
        model = get_cu_hxr_bmad_model(**options)

    assert len(calls) == 1
    assert model.tao.tao_global()["track_type"] == "beam"
    assert model.tao.tao_global()["lattice_calc_on"]
    expected_count = {"custom": 9, "default": 2000, "reference": 1}[source]
    assert model.initial_particles.n_particle == expected_count
    assert model.final_particles.n_particle == expected_count
    assert model.get_value("OTR4_beam").n_particle == expected_count
    if source == "reference":
        reference = model.tao.ele_orbit("BEGINNING")
        for name in ("x", "y", "z", "t", "px", "py"):
            np.testing.assert_array_equal(calls[0][name], [0.0])
            np.testing.assert_allclose(model.initial_particles[name], [0.0], atol=1e-15)
        np.testing.assert_allclose(calls[0].pz, [reference["p0c"]])
        assert calls[0].species == reference["species"].lower()
        assert calls[0].charge == pytest.approx(5e-12)
        np.testing.assert_array_equal(calls[0].status, [1])
    else:
        filename = "test_beam.h5" if source == "custom" else "bmad_set_beam2000_pg.h5"
        assert calls[0] == ParticleGroup(BEAM_DIR / filename)


def test_factory_does_not_load_particles_without_tracking(monkeypatch):
    original = LUMEBmadModel.initial_particles

    def unexpected_set(self, particles):
        pytest.fail("No particle assignment expected when tracking is disabled")

    monkeypatch.setattr(
        LUMEBmadModel, "initial_particles", property(original.fget, unexpected_set)
    )
    model = get_cu_hxr_bmad_model(
        end_element="OTR4", track_beam=False, custom_beam_path="/missing/beam.h5"
    )
    assert model.initial_particles is None


def test_factory_rejects_ascii_custom_beam():
    with pytest.raises(OSError):
        get_cu_hxr_bmad_model(
            end_element="OTR4",
            track_beam=True,
            custom_beam_path=str(BEAM_DIR / "test_beam"),
        )
