"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from trainer.baseline_trainer import build_baseline_trainer
from trainer.build import TRAINER_REGISTRY, build_trainer
from trainer.transformer_trainer import build_transformer_trainer
