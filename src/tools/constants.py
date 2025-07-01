"""Constants."""

# standard library imports
from enum import Enum

# related third party imports
import optuna

# local application/library specific imports
# /

WRITE_DIR = "../data/processed"
ROUND_FLOAT = 4

RACE_PP = "race_pp"
RACE_PP_4K = "race_pp_4000"
RACE_PP_8K = "race_pp_8000"
RACE_PP_12K = "race_pp_12000"
RACE_4K = "race_4000"
RACE_PP_BALANCED = "race_pp_balanced"
ARC = "arc"
ARC_BALANCED = "arc_balanced"
CUPA = "cupa"
AM = "am"
DEV = "dev"
VALIDATION = "validation"
TEST = "test"
TRAIN = "train"
CORRECT_ANSWERS_LIST = "correct_answers_list"
PRED_DIFFICULTY = "predicted_difficulty"
TF_QUESTION_ID = "question_id"
TF_DIFFICULTY = "difficulty"
TF_TEXT = "text"
TF_LABEL = "label"
TF_DESCRIPTION = "description"
TF_ANSWERS = "answers"
TF_CORRECT = "correct"
TF_ANS_ID = "id"

CORRECT_ANSWER = "correct_answer"
OPTIONS = "options"
OPTION_ = "option_"
OPTION_0 = "option_0"
OPTION_1 = "option_1"
OPTION_2 = "option_2"
OPTION_3 = "option_3"
QUESTION = "question"
CONTEXT = "context"
CONTEXT_ID = "context_id"
Q_ID = "q_id"
SPLIT = "split"
DIFFICULTY = "difficulty"
DF_COLS = [
    CORRECT_ANSWER,
    OPTIONS,
    OPTION_0,
    OPTION_1,
    OPTION_2,
    OPTION_3,
    QUESTION,
    CONTEXT,
    CONTEXT_ID,
    Q_ID,
    SPLIT,
    DIFFICULTY,
]

METRIC_BEST_MODEL = {  # NOTE: metrics
    "regression": "eval_bal_rps",
    "classification": "eval_bal_rps",
    "ordinal_coral": "eval_bal_rps",
    "ordinal_corn": "eval_bal_rps",
    "ordinal_or_nn": "eval_bal_rps",
    "ordinal_logit": "eval_bal_rps",
}
GREATER_IS_BETTER = {  # NOTE: metrics
    "eval_rmse": False,
    "eval_accuracy": True,
    "eval_c_idx": True,
    "eval_bal_rmse": False,
    "eval_rps": False,
    "eval_bal_rps": False,
}
OPTUNA_STUDY_DIRECTION = {  # NOTE: metrics
    "eval_rmse": optuna.study.StudyDirection.MINIMIZE,
    "eval_accuracy": optuna.study.StudyDirection.MAXIMIZE,
    "eval_c_idx": optuna.study.StudyDirection.MAXIMIZE,
    "eval_bal_rmse": optuna.study.StudyDirection.MINIMIZE,
    "eval_rps": optuna.study.StudyDirection.MINIMIZE,
    "eval_bal_rps": optuna.study.StudyDirection.MINIMIZE,
}


class OutputType(str, Enum):
    """Neural network output types."""

    REGR = "regression"
    CLASS = "classification"
    ORD_CORAL = "ordinal_coral"
    ORD_CORN = "ordinal_corn"
    ORD_OR_NN = "ordinal_or_nn"
    ORD_LOGIT = "ordinal_logit"


class OrdinalType(str, Enum):
    """Ordinal regression types."""

    CORAL = "ordinal_coral"
    CORN = "ordinal_corn"
    OR_NN = "ordinal_or_nn"
    LOGIT = "ordinal_logit"


class BaselineType(str, Enum):
    """Baseline types."""

    RANDOM = "random"
    MAJORITY = "majority"


MODEL_NAME_SHORT = {
    "distilbert-base-uncased": "distilbert",
    "bert-base-uncased": "bert",
    "answerdotai/ModernBERT-base": "modernbert",
    "majority": "majority",
    "random": "random",
}

EXCLUDE_METRICS = [
    "train_loss",
    "epoch",
    "epochs_trained",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "total_flos",
    "test_samples_per_second",
    "test_steps_per_second",
    "test_runtime",
    "test_acc_class",
    "test_loss",
    "val_acc_class",
    "val_loss",
    "val_rmse",
    "val_mae",
    "val_accuracy",
    "val_f1",
    "val_bal_accuracy",
    "val_bal_f1",
    "val_runtime",
    "val_samples_per_second",
    "val_steps_per_second",
    "val_c_idx",
    "val_bal_rmse",
    "val_rps",
    "val_rps_discrete",
    "val_bal_rps",
    "val_bal_rps_discrete",
]

SKIP_ANSWERS_TEXTS = {
    "race_pp": False,
    "arc": False,
    "cupa": False,
    "am": True,
}
