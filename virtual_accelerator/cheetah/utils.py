"""Cheetah utility helpers that are independent from action variable logic.

The action conversion and PV-mapping layer now lives in
``virtual_accelerator.cheetah.actions``. This module keeps only static mapping
helpers used to load MAD/controls naming tables from CSV.
"""

import os
from pathlib import Path

import pandas as pd

LCLS_ELEMENTS = os.path.join(Path(__file__).parent.resolve(), "lcls_elements.csv")


def get_mad_control_mapping(fname: str | None = None):
    """
    Create a mapping from MAD element names to control-system names.

    Parameters
    ----------
    fname : str | None
        Optional path to a CSV file containing ``Element`` and
        ``Control System Name`` columns.

    """
    if fname is None:
        fname = str(LCLS_ELEMENTS)
    mapping = (
        pd.read_csv(fname, dtype=str)
        .set_index("Element")["Control System Name"]
        .to_dict()
    )
    return mapping


def get_control_mad_mapping(fname: str | None = None):
    """
    Create a mapping from control-system names to MAD element names.

    Parameters
    ----------
    fname : str | None
        Optional path to a CSV file containing ``Control System Name`` and
        ``Element`` columns.

    """

    if fname is None:
        fname = str(LCLS_ELEMENTS)

    mapping = (
        pd.read_csv(fname, dtype=str)
        .set_index("Control System Name")["Element"]
        .T.to_dict()
    )
    return mapping
