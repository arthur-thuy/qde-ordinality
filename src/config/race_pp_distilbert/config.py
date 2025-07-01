"""Default config file."""

from yacs.config import CfgNode as CN

_C = CN()

# seed for reproducibility
_C.SEED = 42

# number of runs
_C.RUNS = 5  # 1  # 5

_C.LOADER = CN()
# dataset name
_C.LOADER.NAME = "race_pp"
# number of data loading workers
# _C.LOADER.WORKERS = 4 # NOTE: unused because warning
# output type
_C.LOADER.OUTPUT_TYPE = "classification"
# use smaller dev set
_C.LOADER.SMALL_DEV = None
# use balanced version
_C.LOADER.BALANCED = False
# discretize labels
_C.LOADER.DISCRETIZER = None

# model architecture
_C.MODEL = CN()
# model name
_C.MODEL.NAME = "distilbert-base-uncased"
# number of labels (1 for regression)
_C.MODEL.NUM_LABELS = 3
# max sequence length
_C.MODEL.MAX_LENGTH = 512  # 256
# BF16
_C.MODEL.BF16 = True
# TF32
_C.MODEL.TF32 = True
# torch compile
_C.MODEL.TORCH_COMPILE = False


_C.TRAIN = CN()
# number of total epochs to run
_C.TRAIN.EPOCHS = 3
# max number of training steps
_C.TRAIN.MAX_STEPS = -1  # 250
# mini-batch size
_C.TRAIN.BATCH_SIZE = 128
# initial learning rate
_C.TRAIN.LR = 2e-5
# weight decay
_C.TRAIN.WEIGHT_DECAY = 0.05
# Adam epsilon
_C.TRAIN.ADAM_EPSILON = 1e-6  # NOTE: 1e-6 is default in HF
# freeze base
_C.TRAIN.FREEZE_BASE = False
# warmup ratio
_C.TRAIN.WARMUP_RATIO = 0.1
# early stopping
_C.TRAIN.EARLY_STOPPING = True
# patience for early stopping
_C.TRAIN.PATIENCE = 20  # NOTE: 10 is too short if 0.01 logging steps

_C.EVAL = CN()
# batch size
_C.EVAL.BATCH_SIZE = 256
# logging and eval steps
_C.EVAL.LOGGING_STEPS = 0.02  # 0.01

_C.DEVICE = CN()
# disables CUDA training
_C.DEVICE.NO_CUDA = False
# disables macOS GPU training
_C.DEVICE.NO_MPS = False


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
