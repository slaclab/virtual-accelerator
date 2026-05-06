import pytest

from virtual_accelerator.utils.optional_dependencies import (
    import_optional,
    import_optional_symbol,
)


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


def test_import_optional_propagates_nested_module_not_found(monkeypatch):
    def fake_import_module(_name):
        raise ModuleNotFoundError("No module named 'lume_cheetah'", name="lume_cheetah")

    monkeypatch.setattr(
        "virtual_accelerator.utils.optional_dependencies.import_module",
        fake_import_module,
    )

    with pytest.raises(ModuleNotFoundError):
        import_optional(
            "virtual_accelerator.cheetah.transformer",
            feature="test feature",
            extra="cheetah",
        )


def test_import_optional_wraps_missing_requested_module(monkeypatch):
    def fake_import_module(_name):
        raise ModuleNotFoundError("No module named 'cheetah'", name="cheetah")

    monkeypatch.setattr(
        "virtual_accelerator.utils.optional_dependencies.import_module",
        fake_import_module,
    )

    with pytest.raises(ImportError) as exc_info:
        import_optional(
            "cheetah.accelerator",
            feature="test feature",
            extra="cheetah",
        )

    message = str(exc_info.value)
    assert "test feature requires optional dependency 'cheetah.accelerator'" in message
