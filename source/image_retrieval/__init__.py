"""Image retrieval backend package.

This package contains the retrieval API, data download utilities, and model
helpers used by the README workflow. The original script-based entry points are
kept working for direct execution while the package can also be imported as
`source.image_retrieval`.
"""

__all__ = [
    "common",
    "download_data",
    "search_api",
    "serve",
]
