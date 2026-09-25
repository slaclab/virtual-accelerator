"""Registry for cached beam distributions under ``virtual_accelerator/beams/``.

Each ``.h5`` beam distribution ships with a matching ``.meta.json`` sidecar
describing which beamline it belongs to, which element it was captured at, the
operating mode, and other physical parameters. This module discovers those
sidecars lazily on first use and exposes two entry points:

* :func:`get_beam` — load beams by ``beamline``, ``element``, and ``mode``.
* :func:`list_beams` — enumerate metadata entries without opening the HDF5.

The registry walks the beams directory the first time it is queried and caches
the result for the lifetime of the process. Call :func:`clear_cache` in tests
that mutate the beams directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from virtual_accelerator.utils.optional_dependencies import import_optional_symbol


BEAMS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BeamEntry:
    """One cached beam distribution and its sidecar metadata.

    ``path`` points at the ``.h5`` file; the remaining fields mirror the
    ``.meta.json`` schema documented in the top-level README. Use
    :meth:`load` to materialize the beam as an openPMD ``ParticleGroup``.
    """

    path: Path
    beamline: str
    element: str
    mode: str
    s_m: float
    ref_energy_eV: float
    generator: str
    n_particles: int
    charge_C: float
    species: str
    date_generated: str
    source: str
    notes: str
    extra: dict[str, Any] = field(default_factory=dict)

    def load(self):
        """Return the beam as a ``pmd_beamphysics.ParticleGroup``.

        Handles two on-disk layouts:

        * root layout — particle datasets at ``/position/x`` etc.
        * openPMD tree — datasets under ``/particles/<species>/``.
        """
        ParticleGroup = import_optional_symbol(
            "pmd_beamphysics",
            "ParticleGroup",
            feature="loading cached beam distributions",
            extra="bmad",
        )
        import h5py

        with h5py.File(str(self.path), "r") as f:
            if "particles" in f:
                species_names = list(f["particles"].keys())
                if not species_names:
                    raise ValueError(f"{self.path} has an empty /particles group")
                # Prefer the sidecar's species when it's present; otherwise
                # take the only species we find.
                species = (
                    self.species if self.species in species_names else species_names[0]
                )
                return ParticleGroup(h5=f[f"particles/{species}"])
            return ParticleGroup(h5=f)


_KNOWN_FIELDS = {
    "beamline",
    "element",
    "mode",
    "s_m",
    "ref_energy_eV",
    "generator",
    "n_particles",
    "charge_C",
    "species",
    "date_generated",
    "source",
    "notes",
}


_cache: list[BeamEntry] | None = None


def _scan() -> list[BeamEntry]:
    entries: list[BeamEntry] = []
    for sidecar in sorted(BEAMS_DIR.rglob("*.meta.json")):
        data = json.loads(sidecar.read_text())
        # Strip the ".meta.json" double-suffix to get the paired .h5 path.
        h5 = sidecar.with_name(sidecar.name.removesuffix(".meta.json") + ".h5")
        if not h5.exists():
            # Sidecar without a beam file — skip; the pre-commit hook and
            # test_beam_sidecars would already flag this.
            continue
        known = {k: data[k] for k in _KNOWN_FIELDS if k in data}
        extra = {k: v for k, v in data.items() if k not in _KNOWN_FIELDS}
        entries.append(BeamEntry(path=h5, extra=extra, **known))
    return entries


def _entries() -> list[BeamEntry]:
    global _cache
    if _cache is None:
        _cache = _scan()
    return _cache


def clear_cache() -> None:
    """Force the next :func:`list_beams`/:func:`get_beam` call to re-scan."""
    global _cache
    _cache = None


def list_beams(
    *,
    beamline: str | None = None,
    element: str | None = None,
    mode: str | None = None,
) -> list[BeamEntry]:
    """Return metadata entries matching the given filters.

    Filters are ANDed together. Passing ``None`` (the default) skips that
    filter. Does not open the ``.h5`` blobs.
    """
    results = _entries()
    if beamline is not None:
        results = [e for e in results if e.beamline == beamline]
    if element is not None:
        results = [e for e in results if e.element == element]
    if mode is not None:
        results = [e for e in results if e.mode == mode]
    return results


def get_beam(
    beamline: str,
    element: str,
    mode: str,
):
    """Return every cached beam matching ``(beamline, element, mode)``.

    Raises ``KeyError`` with the list of available ``(element, mode)`` pairs
    for that beamline when nothing matches.
    """
    matches = list_beams(beamline=beamline, element=element, mode=mode)
    if not matches:
        available = sorted({(e.element, e.mode) for e in list_beams(beamline=beamline)})
        if not available:
            raise KeyError(
                f"No beams cached for beamline={beamline!r}. "
                f"Known beamlines: {sorted({e.beamline for e in _entries()})}"
            )
        raise KeyError(
            f"No beam for beamline={beamline!r}, element={element!r}, "
            f"mode={mode!r}. Available (element, mode) for {beamline!r}: "
            f"{available}"
        )
    return [entry.load() for entry in matches]


__all__ = [
    "BeamEntry",
    "BEAMS_DIR",
    "list_beams",
    "get_beam",
    "clear_cache",
]
