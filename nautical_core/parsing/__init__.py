"""Parser domain implementations.

The historical ``nautical_core.parser_*`` modules remain compatibility
shims; new internal code should import parser components from this package.
"""

from . import parser_atoms, parser_dnf, parser_frontend, parser_models, parser_support_api

__all__ = (
    "parser_atoms",
    "parser_dnf",
    "parser_frontend",
    "parser_models",
    "parser_support_api",
)
