"""Tuning module."""

# standard library imports
# /

# related third party imports
import optuna
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from tools.configurator import create_config_id
from tuner.build import TUNER_REGISTRY

logger = structlog.get_logger(__name__)


@TUNER_REGISTRY.register("race_pp_distilbert_regression")
def build_race_pp_distilbert_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_distilbert_classification")
def build_race_pp_distilbert_classification(
    cfg: CfgNode, trial: optuna.Trial
) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_distilbert_ordinal_or_nn")
def build_race_pp_distilbert_ordinal_or_nn(
    cfg: CfgNode, trial: optuna.Trial
) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_distilbert_ordinal_coral")
def build_race_pp_distilbert_ordinal_coral(
    cfg: CfgNode, trial: optuna.Trial
) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_distilbert_ordinal_corn")
def build_race_pp_distilbert_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_distilbert_regression")
def build_arc_distilbert_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_distilbert_classification")
def build_arc_distilbert_classification(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_distilbert_ordinal_or_nn")
def build_arc_distilbert_ordinal_or_nn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_distilbert_ordinal_coral")
def build_arc_distilbert_ordinal_coral(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_distilbert_ordinal_corn")
def build_arc_distilbert_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=1e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=5e-4, high=1e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_bert_regression")
def build_arc_bert_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_bert_classification")
def build_arc_bert_classification(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_bert_ordinal_or_nn")
def build_arc_bert_ordinal_or_nn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_bert_ordinal_coral")
def build_arc_bert_ordinal_coral(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_bert_ordinal_corn")
def build_arc_bert_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_bert_ordinal_logit")
def build_arc_bert_ordinal_logit(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_bert_regression")
def build_race_pp_bert_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_bert_classification")
def build_race_pp_bert_classification(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_bert_ordinal_or_nn")
def build_race_pp_bert_ordinal_or_nn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_bert_ordinal_coral")
def build_race_pp_bert_ordinal_coral(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_bert_ordinal_corn")
def build_race_pp_bert_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_bert_ordinal_logit")
def build_race_pp_bert_ordinal_logit(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_modernbert_regression")
def build_race_pp_modernbert_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_modernbert_classification")
def build_race_pp_modernbert_classification(
    cfg: CfgNode, trial: optuna.Trial
) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_modernbert_ordinal_or_nn")
def build_race_pp_modernbert_ordinal_or_nn(
    cfg: CfgNode, trial: optuna.Trial
) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_modernbert_ordinal_coral")
def build_race_pp_modernbert_ordinal_coral(
    cfg: CfgNode, trial: optuna.Trial
) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_modernbert_ordinal_corn")
def build_race_pp_modernbert_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_modernbert_regression")
def build_arc_modernbert_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_modernbert_classification")
def build_arc_modernbert_classification(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_modernbert_ordinal_or_nn")
def build_arc_modernbert_ordinal_or_nn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_modernbert_ordinal_coral")
def build_arc_modernbert_ordinal_coral(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("arc_modernbert_ordinal_corn")
def build_arc_modernbert_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_roberta_regression")
def build_race_pp_roberta_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_roberta_classification")
def build_race_pp_roberta_classification(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_roberta_ordinal_or_nn")
def build_race_pp_roberta_ordinal_or_nn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_roberta_ordinal_coral")
def build_race_pp_roberta_ordinal_coral(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("race_pp_roberta_ordinal_corn")
def build_race_pp_roberta_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("am_bert_regression")
def build_am_bert_regression(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("am_bert_classification")
def build_am_bert_classification(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("am_bert_ordinal_or_nn")
def build_am_bert_ordinal_or_nn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("am_bert_ordinal_coral")
def build_am_bert_ordinal_coral(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg


@TUNER_REGISTRY.register("am_bert_ordinal_corn")
def build_am_bert_ordinal_corn(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    # NOTE: better to define float -> TPE recognizes ordering in numbers

    # clone cfg
    tuner_cfg = cfg.clone()

    # lr
    lr_suggest = trial.suggest_float("lr", low=1e-5, high=5e-4, log=True)
    tuner_cfg.TRAIN.LR = lr_suggest
    assert tuner_cfg.TRAIN.LR == lr_suggest

    # weight decay
    weight_decay_suggest = trial.suggest_float(
        "weight_decay", low=1e-3, high=5e-1, log=True
    )
    tuner_cfg.TRAIN.WEIGHT_DECAY = weight_decay_suggest
    assert tuner_cfg.TRAIN.WEIGHT_DECAY == weight_decay_suggest

    # redo ID with updated hyperparams
    tuner_cfg.ID = create_config_id(tuner_cfg)

    return tuner_cfg
