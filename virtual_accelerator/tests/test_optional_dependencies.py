import pytest

from virtual_accelerator.utils.optional_dependencies import import_optional_symbol


def test_missing_optional_dependency_error_message():
    with pytest.raises(ImportError) as exc_info:
        import_optional_symbol(
            "virtual_accelerator.this_module_does_not_exist",
            "anything",
            feature="test feature",
            extra="cheetah",
        )

    message = str(exc_info.value)
    assert "test feature requires optional dependency" in message
    assert "pip install virtual-accelerator[cheetah]" in message
