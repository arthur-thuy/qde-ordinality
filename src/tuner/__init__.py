"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from tuner.build import TUNER_REGISTRY, build_tuner_cfg
from tuner.tuner import (
    build_race_pp_distilbert_regression,
)
