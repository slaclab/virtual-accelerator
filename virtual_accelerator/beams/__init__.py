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
        """Materialize the on-disk beam as a ``pmd_beamphysics.ParticleGroup``.

        Opens ``self.path`` with h5py and hands the relevant group to
        ``ParticleGroup``. Two on-disk layouts are supported transparently:

        * **root layout** — particle datasets live at the file root
          (``/position/x``, ``/momentum/px``, ...). The entire file handle is
          passed straight through.
        * **openPMD tree** — datasets live under ``/particles/<species>/``.
          The species named in this entry's sidecar is preferred; if it is
          not present in the file, the first (and typically only) species
          group is used instead.

        Returns
        -------
        pmd_beamphysics.ParticleGroup
            The loaded beam distribution.

        Raises
        ------
        ImportError
            If ``pmd_beamphysics`` is not installed. Install the ``bmad``
            extra to pull it in.
        ValueError
            If the file contains an openPMD ``/particles`` group with no
            species inside.
        OSError
            If ``self.path`` cannot be opened by h5py.
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
    """Discard the in-memory registry so the next lookup re-scans the disk.

    The registry walks ``BEAMS_DIR`` once and memoizes the result for the
    lifetime of the process, which is fine for normal use but stale in tests
    that add, remove, or rewrite beam files at runtime. Call this after any
    such mutation so the next :func:`list_beams` or :func:`get_beam` call
    sees the new state.

    Returns
    -------
    None
    """
    global _cache
    _cache = None


def list_beams(
    *,
    beamline: str | None = None,
    element: str | None = None,
    mode: str | None = None,
) -> list[BeamEntry]:
    """Return cached :class:`BeamEntry` records matching the given filters.

    All filters are keyword-only and ANDed together; passing ``None`` (the
    default) skips that filter and matches every value. Only the sidecar
    metadata is consulted — the ``.h5`` blobs are never opened, so this call
    is cheap and safe to use for discovery, listing, or completion.

    Parameters
    ----------
    beamline : str, optional
        Match only entries whose ``beamline`` field equals this value.
    element : str, optional
        Match only entries whose ``element`` field equals this value.
    mode : str, optional
        Match only entries whose ``mode`` field equals this value.

    Returns
    -------
    list[BeamEntry]
        Matching entries in the order they were discovered on disk. Empty
        list when nothing matches — this function never raises for a miss;
        use :func:`get_beam` if you want an error instead.

    Examples
    --------
    >>> list_beams(beamline="FACET2")                  # doctest: +SKIP
    >>> list_beams(beamline="FACET2", element="L0AFEND")  # doctest: +SKIP
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
    """Load every cached beam matching ``(beamline, element, mode)``.

    Looks up sidecars via :func:`list_beams` and materializes each match by
    calling :meth:`BeamEntry.load`. All three keys are required — this is
    the strict, "give me the beam or fail loudly" entry point; use
    :func:`list_beams` for filtered discovery that never raises.

    Parameters
    ----------
    beamline : str
        Beamline identifier from the sidecar (e.g. ``"FACET2"``).
    element : str
        Element identifier at which the beam was captured
        (e.g. ``"L0AFEND"``).
    mode : str
        Operating mode label from the sidecar (e.g. ``"oneBunch"``).

    Returns
    -------
    list[pmd_beamphysics.ParticleGroup]
        One entry per matching sidecar, in discovery order. Multiple hits
        are possible when several cached beams share the same
        ``(beamline, element, mode)`` triple (for example, different
        particle counts or generation dates).

    Raises
    ------
    KeyError
        If no cached beam matches. The message enumerates the available
        ``(element, mode)`` pairs for that beamline, or the known beamlines
        when the beamline itself is unknown.
    ImportError
        Propagated from :meth:`BeamEntry.load` when ``pmd_beamphysics`` is
        not installed.
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
