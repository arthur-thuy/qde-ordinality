"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from discretizer.build import DISCRETIZER_REGISTRY, build_discretizer
from discretizer.discretizer import (
    build_equal_width,
    build_variable_width_right,
    build_variable_width_symmetric,
)