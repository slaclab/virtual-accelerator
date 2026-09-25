"""Tests for the ``virtual_accelerator.beams`` registry."""

import json

import pytest

from virtual_accelerator import beams as beams_module
from virtual_accelerator.beams import (
    BEAMS_DIR,
    BeamEntry,
    clear_cache,
    list_beams,
    get_beam,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


def test_scan_discovers_every_on_disk_sidecar():
    on_disk = sorted(p.name for p in BEAMS_DIR.rglob("*.meta.json"))
    entries = list_beams()
    discovered = sorted(e.path.name.replace(".h5", ".meta.json") for e in entries)
    assert discovered == on_disk, (
        f"Registry scan missed sidecars. On disk: {on_disk}, discovered: {discovered}"
    )


def test_list_beams_filters_by_beamline():
    facet_beams = list_beams(beamline="facet2")
    assert facet_beams, "expected at least one facet2 beam in the cache"
    assert all(e.beamline == "facet2" for e in facet_beams)


def test_list_beams_filters_are_anded():
    entries = list_beams(beamline="facet2", element="PR10241", mode="nominal_one_bunch")
    assert entries
    for e in entries:
        assert e.beamline == "facet2"
        assert e.element == "PR10241"
        assert e.mode == "nominal_one_bunch"


def test_list_beams_returns_empty_list_when_nothing_matches():
    assert list_beams(beamline="does_not_exist") == []


def test_get_beam_raises_with_available_options_when_element_missing():
    with pytest.raises(KeyError) as excinfo:
        get_beam(beamline="facet2", element="NOPE", mode="nominal_one_bunch")
    msg = str(excinfo.value)
    assert "facet2" in msg
    assert "NOPE" in msg
    # The error must list at least one real (element, mode) pair for the beamline.
    assert "PR10241" in msg or "L0AFEND" in msg


def test_get_beam_raises_when_beamline_unknown():
    with pytest.raises(KeyError) as excinfo:
        get_beam(beamline="nonexistent", element="X", mode="Y")
    assert "Known beamlines" in str(excinfo.value)


def test_get_beam_returns_list_matching_list_beams():
    """get_beam should return one entry per list_beams match."""
    pytest.importorskip("pmd_beamphysics")
    matches = list_beams(beamline="facet2", element="L0AFEND", mode="nominal_one_bunch")
    if not matches:
        pytest.skip("no L0AFEND beams cached")
    # Skip if any matching .h5 is an unresolved LFS pointer stub.
    for entry in matches:
        if entry.path.stat().st_size < 1024:
            head = entry.path.read_bytes()[:64]
            if head.startswith(b"version https://git-lfs"):
                pytest.skip(f"{entry.path.name} is an unresolved Git LFS pointer")
    beams_list = get_beam(
        beamline="facet2", element="L0AFEND", mode="nominal_one_bunch"
    )
    assert isinstance(beams_list, list)
    assert len(beams_list) == len(matches)


def test_cache_is_reused_across_calls():
    first = list_beams()
    second = list_beams()
    # Same underlying list object — no re-scan.
    assert beams_module._cache is not None
    assert first is second


def test_clear_cache_forces_rescan():
    list_beams()
    assert beams_module._cache is not None
    clear_cache()
    assert beams_module._cache is None
    list_beams()
    assert beams_module._cache is not None


def test_beam_entry_captures_unknown_sidecar_fields(tmp_path, monkeypatch):
    """Fields beyond the known schema land in `extra` instead of erroring."""
    fake_beams = tmp_path / "beams"
    (fake_beams / "scenario").mkdir(parents=True)
    h5 = fake_beams / "scenario" / "TEST_100.h5"
    h5.write_bytes(b"not a real h5 but the registry doesn't open it")
    sidecar = fake_beams / "scenario" / "TEST_100.meta.json"
    sidecar.write_text(
        json.dumps(
            {
                "beamline": "test_line",
                "element": "TEST",
                "mode": "nominal",
                "s_m": 0.0,
                "ref_energy_eV": 1e6,
                "generator": "synthetic",
                "n_particles": 100,
                "charge_C": 1e-12,
                "species": "electron",
                "date_generated": "2026-01-01",
                "source": "unit test",
                "notes": "",
                "future_field": "hello",
            }
        )
    )
    monkeypatch.setattr(beams_module, "BEAMS_DIR", fake_beams)
    clear_cache()
    entries = list_beams(beamline="test_line")
    assert len(entries) == 1
    assert isinstance(entries[0], BeamEntry)
    assert entries[0].extra == {"future_field": "hello"}
