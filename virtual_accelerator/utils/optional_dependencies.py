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
    return getattr(module, symbol_name)