"""Kiln runtime package."""

try:
    from kiln._kiln import __version__ as __version__
except ImportError:  # pragma: no cover - extension not built yet
    __version__ = "0.1.0"


def main() -> None:
    print(f"kiln {__version__}")
