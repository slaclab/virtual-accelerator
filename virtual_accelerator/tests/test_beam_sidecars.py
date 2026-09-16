"""Structural checks on cached beam files under ``virtual_accelerator/beams/``.

Every ``.h5`` beam distribution must ship with a matching ``.json`` sidecar of
the same basename, and every sidecar must contain a minimal set of fields so a
reader can tell what the beam represents without opening the (possibly LFS-only)
HDF5 blob.
"""

import json
from pathlib import Path

import pytest


BEAMS_DIR = Path(__file__).resolve().parent.parent / "beams"

REQUIRED_SIDECAR_FIELDS = frozenset(
    {
        "plane",
        "s_m",
        "ref_energy_eV",
        "generator",
        "n_particles",
        "date_generated",
        "mode",
        "source",
    }
)


def _h5_files() -> list[Path]:
    return sorted(BEAMS_DIR.rglob("*.h5"))


@pytest.mark.parametrize(
    "h5_path", _h5_files(), ids=lambda p: str(p.relative_to(BEAMS_DIR))
)
def test_h5_beam_has_json_sidecar(h5_path: Path) -> None:
    sidecar = h5_path.with_suffix(".json")
    assert sidecar.exists(), (
        f"{h5_path.relative_to(BEAMS_DIR)} has no sidecar. "
        f"Expected {sidecar.name} alongside it."
    )


@pytest.mark.parametrize(
    "h5_path", _h5_files(), ids=lambda p: str(p.relative_to(BEAMS_DIR))
)
def test_sidecar_has_required_fields(h5_path: Path) -> None:
    sidecar = h5_path.with_suffix(".json")
    if not sidecar.exists():
        pytest.skip("sidecar missing; covered by test_h5_beam_has_json_sidecar")
    data = json.loads(sidecar.read_text())
    missing = REQUIRED_SIDECAR_FIELDS - data.keys()
    assert not missing, (
        f"{sidecar.relative_to(BEAMS_DIR)} is missing required fields: "
        f"{sorted(missing)}"
    )
