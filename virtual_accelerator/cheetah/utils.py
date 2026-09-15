"""Cheetah utility helpers that are independent from action variable logic.

The action conversion and PV-mapping layer now lives in
``virtual_accelerator.cheetah.actions``. This module keeps only static mapping
helpers used to load MAD/controls naming tables from CSV.

The elements table is not bundled with this package -- it lives in the lattice
repository, at ``$LCLS_LATTICE/bmad/conversion/from_oracle/lcls_elements.csv``.
These helpers therefore take an explicit path, and resolving it (including
reporting an unset ``LCLS_LATTICE``) is the caller's job, as in
``virtual_accelerator.utils.variables``.
"""

import pandas as pd


def _read_lcls_elements(fname: str) -> pd.DataFrame:
    """Read the LCLS elements CSV, tolerating an optional category header row.

    Newer lattice exports prepend a grouping row (e.g. ``EPICS Channel Access
    Device``) above the real column header. Detect that case and use the second
    row as the header instead.
    """
    frame = pd.read_csv(fname, dtype=str)
    if "Element" not in frame.columns:
        frame = pd.read_csv(fname, dtype=str, header=1)
    return frame


def get_mad_control_mapping(fname: str):
    """
    Create a mapping from MAD element names to control-system names.

    Parameters
    ----------
    fname : str
        Path to a CSV file containing ``Element`` and ``Control System Name``
        columns, e.g. ``$LCLS_LATTICE/bmad/conversion/from_oracle/lcls_elements.csv``.
        The table is not bundled with this package, so the path is required.

    Returns
    -------
    dict
        Mapping of MAD element name -> control-system PV prefix.

    """
    mapping = (
        _read_lcls_elements(fname).set_index("Element")["Control System Name"].to_dict()
    )
    return mapping


def get_control_mad_mapping(fname: str):
    """
    Create a mapping from control-system names to MAD element names.

    Parameters
    ----------
    fname : str
        Path to a CSV file containing ``Control System Name`` and ``Element``
        columns, e.g.
        ``$LCLS_LATTICE/bmad/conversion/from_oracle/lcls_elements.csv``. The
        table is not bundled with this package, so the path is required.

    Returns
    -------
    dict
        Mapping of control-system PV prefix -> MAD element name.

    """
    mapping = (
        _read_lcls_elements(fname)
        .set_index("Control System Name")["Element"]
        .T.to_dict()
    )
    return mapping
