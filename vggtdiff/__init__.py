"""Public package. The generation model is imported only when requested."""

__all__ = ["VGGTDiff"]


def __getattr__(name):
    if name == "VGGTDiff":
        from .model import VGGTDiff

        return VGGTDiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
