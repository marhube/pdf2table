# For Python 3.8 and later
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("pdf2table")
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'