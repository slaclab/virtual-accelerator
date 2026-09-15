import pytest

from virtual_accelerator.cheetah.utils import (
    get_control_mad_mapping,
    get_mad_control_mapping,
)

# Synthetic tables: these helpers only pair two named columns, so the element and
# PV names below are arbitrary and no lattice is involved. The real table has ~60
# more columns and covers several beampaths.
ELEMENTS_CSV = """Element,Control System Name
Q1,QUAD:IN10:511
PR1,OTRS:IN10:571
"""

# Same transform, different naming convention -- nothing here is facility-specific.
FACET_STYLE_CSV = """Element,Control System Name
QE10525,QUAD:IN10:525
PR10571,PROF:IN10:571
"""

# Newer lattice exports prepend a grouping row above the real column header.
ELEMENTS_CSV_WITH_CATEGORY_ROW = """EPICS Channel Access Device,EPICS Channel Access Device
Element,Control System Name
Q1,QUAD:IN10:511
PR1,OTRS:IN10:571
"""


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "lcls_elements.csv"
    path.write_text(ELEMENTS_CSV)
    return path


def test_mad_control_mapping(database_path):
    mapping = get_mad_control_mapping(str(database_path))

    assert mapping == {"Q1": "QUAD:IN10:511", "PR1": "OTRS:IN10:571"}


def test_names_pass_through_verbatim(tmp_path):
    # Whatever the naming convention, both columns are carried through unchanged:
    # these helpers never inspect or normalize the names.
    path = tmp_path / "lcls_elements.csv"
    path.write_text(FACET_STYLE_CSV)

    assert get_mad_control_mapping(str(path)) == {
        "QE10525": "QUAD:IN10:525",
        "PR10571": "PROF:IN10:571",
    }


def test_control_mad_mapping_inverts(database_path):
    forward = get_mad_control_mapping(str(database_path))
    reverse = get_control_mad_mapping(str(database_path))

    assert reverse == {v: k for k, v in forward.items()}


def test_category_header_row_is_tolerated(tmp_path):
    path = tmp_path / "lcls_elements.csv"
    path.write_text(ELEMENTS_CSV_WITH_CATEGORY_ROW)

    assert get_mad_control_mapping(str(path)) == {
        "Q1": "QUAD:IN10:511",
        "PR1": "OTRS:IN10:571",
    }


@pytest.mark.parametrize("func", [get_mad_control_mapping, get_control_mad_mapping])
def test_path_is_required(func):
    # The table is not bundled, so there is no default to fall back to: omitting
    # the path must fail at the call site rather than chasing a missing file.
    with pytest.raises(TypeError):
        func()


@pytest.mark.parametrize("func", [get_mad_control_mapping, get_control_mad_mapping])
def test_missing_file_raises(func, tmp_path):
    with pytest.raises(FileNotFoundError):
        func(str(tmp_path / "absent.csv"))
