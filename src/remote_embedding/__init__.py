"""Public package exports for remote-embedding."""

from importlib.metadata import PackageNotFoundError, version

from .remote import RemoteEmbeddings

__all__ = ["RemoteEmbeddings"]

try:
    __version__ = version("remote-embedding")
except PackageNotFoundError:
    __version__ = "0.0.0"
