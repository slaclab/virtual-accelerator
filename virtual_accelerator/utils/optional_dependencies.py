from importlib import import_module
from typing import Any


def import_optional(module_name: str, feature: str, extra: str) -> Any:
    """Import an optional module with an actionable error for missing extras."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires optional dependency '{module_name}'. "
            f"Install it with: pip install virtual-accelerator[{extra}]"
        ) from exc


def import_optional_symbol(
    module_name: str, symbol_name: str, feature: str, extra: str
) -> Any:
    """Import a symbol from an optional module with a clear install hint."""
    module = import_optional(module_name, feature=feature, extra=extra)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(
            f"{feature} requires symbol '{symbol_name}' from optional dependency "
            f"'{module_name}', but it was not found. Install it with: "
            f"pip install virtual-accelerator[{extra}]. "
            f"If it is already installed, check that you have a compatible version "
            f"that provides '{symbol_name}'."
        ) from exc
