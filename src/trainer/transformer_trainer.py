"""Module for HuggingFace trainer."""

# standard library imports
import os
import time
from functools import partial
from typing import Any, Callable, Optional

# related third party imports
import structlog
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from torch.optim import AdamW
from yacs.config import CfgNode

# local application/library specific imports
from model.build import build_model
from tools.constants import (
    GREATER_IS_BETTER,
    METRIC_BEST_MODEL,
    ROUND_FLOAT,
    TEST,
    TF_TEXT,
    TRAIN,
    VALIDATION,
    OutputType,
)
from tools.metrics import compute_metrics
from tools.utils import format_time
from trainer.build import TRAINER_REGISTRY

# set up logger
logger = structlog.get_logger(__name__)


@TRAINER_REGISTRY.register("transformer")
def build_transformer_trainer() -> Callable:
    """Build transformer trainer.

    Returns
    -------
    Callable
        Trainer function
    """
    return transformer_trainer


def transformer_trainer(
    cfg: CfgNode,
    datasets,
    device: Any,
    dry_run: bool = False,
    bin_edges: Optional[ArrayLike] = None,
) -> dict:
    """Run HF procedure for single config, single run.

    Parameters
    ----------
    cfg : CfgNode
        Config object
    datasets : _type_
        HF datasets
    device : Any
        Compute device (GPU or CPU)
    dry_run : bool
        Run only one AL epoch, by default False
    bin_edges : Optional[ArrayLike]
        Bin edges for discretization, by default None

    Returns
    -------
    Union[dict[str, Union[int, float, list[Any]]], tuple[dict, list, dict]]
        Output metrics
    """
    # create HF model + tokenizer
    model, tokenizer = build_model(
        model_cfg=cfg.MODEL, output_type=cfg.LOADER.OUTPUT_TYPE
    )
    model = model.to(device)

    if cfg.TRAIN.FREEZE_BASE:
        logger.info("Freezing base model")
        # freeze base model
        for param in model.base_model.parameters():
            param.requires_grad = False
    else:
        logger.info("Fine-tuning base model")

    # tokenize dataset
    def _preprocess_function(examples, tokenizer, max_length):
        return tokenizer(
            examples[TF_TEXT], truncation=True, max_length=max_length, padding=True
        )

    tokenized_dataset = datasets.map(
        _preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.MODEL.MAX_LENGTH},
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create trainer and loop objects
    logging_eval_save_steps = cfg.EVAL.LOGGING_STEPS / cfg.TRAIN.EPOCHS

    if cfg.LOADER.OUTPUT_TYPE != OutputType.ORD_LOGIT:
        training_args = TrainingArguments(
            # logging
            output_dir=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
            overwrite_output_dir=True,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=logging_eval_save_steps,  # train loss calculated per logging_steps # noqa
            save_strategy=IntervalStrategy.STEPS,
            save_steps=logging_eval_save_steps,
            load_best_model_at_end=True,
            metric_for_best_model=METRIC_BEST_MODEL[cfg.LOADER.OUTPUT_TYPE],
            greater_is_better=GREATER_IS_BETTER[
                METRIC_BEST_MODEL[cfg.LOADER.OUTPUT_TYPE]
            ],
            save_total_limit=1,  # keep only the best model
            # convergence
            num_train_epochs=cfg.TRAIN.EPOCHS,
            max_steps=5 if dry_run else cfg.TRAIN.MAX_STEPS,
            learning_rate=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            adam_epsilon=cfg.TRAIN.ADAM_EPSILON,
            per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
            warmup_ratio=cfg.TRAIN.WARMUP_RATIO,
            # optimization
            bf16=cfg.MODEL.BF16,
            tf32=cfg.MODEL.TF32,
            torch_compile=cfg.MODEL.TORCH_COMPILE,
            # evaluation
            per_device_eval_batch_size=cfg.EVAL.BATCH_SIZE,
            eval_strategy=IntervalStrategy.STEPS,
            eval_steps=logging_eval_save_steps,  # eval loss calculated per eval_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset[TRAIN],
            eval_dataset=tokenized_dataset[VALIDATION],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=partial(
                compute_metrics,
                output_type=cfg.LOADER.OUTPUT_TYPE,
                num_classes=cfg.MODEL.NUM_LABELS,
                bin_edges=bin_edges,
            ),
            callbacks=(
                [
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg.TRAIN.PATIENCE,
                        early_stopping_threshold=1e-4,
                    )
                ]
                if cfg.TRAIN.EARLY_STOPPING
                else None
            ),  # NOTE: patience is number of evaluation calls
        )
    else:
        training_args = TrainingArguments(
            # logging
            output_dir=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
            overwrite_output_dir=True,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=logging_eval_save_steps,  # train loss calculated per logging_steps # noqa
            save_strategy=IntervalStrategy.STEPS,
            save_steps=logging_eval_save_steps,
            load_best_model_at_end=True,
            metric_for_best_model=METRIC_BEST_MODEL[cfg.LOADER.OUTPUT_TYPE],
            greater_is_better=GREATER_IS_BETTER[
                METRIC_BEST_MODEL[cfg.LOADER.OUTPUT_TYPE]
            ],
            save_total_limit=1,  # keep only the best model
            # convergence
            num_train_epochs=cfg.TRAIN.EPOCHS,
            max_steps=5 if dry_run else cfg.TRAIN.MAX_STEPS,
            learning_rate=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            adam_epsilon=cfg.TRAIN.ADAM_EPSILON,
            per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
            warmup_ratio=cfg.TRAIN.WARMUP_RATIO,
            # optimization
            bf16=cfg.MODEL.BF16,
            tf32=cfg.MODEL.TF32,
            torch_compile=cfg.MODEL.TORCH_COMPILE,
            # evaluation
            per_device_eval_batch_size=cfg.EVAL.BATCH_SIZE,
            eval_strategy=IntervalStrategy.STEPS,
            eval_steps=logging_eval_save_steps,  # eval loss calculated per eval_steps
        )

        optimizer = create_optimizer_orderedlogit(
            model=model,
            args=training_args,
            lr_multiplier=cfg.TRAIN.LR_MULTIPLIER,
            params_high_lr_names=cfg.TRAIN.PARAMS_HIGH_LR_NAMES,
        )
        lr_scheduler = create_scheduler(
            num_training_steps=training_args.max_steps,
            optimizer=optimizer,
            args=training_args,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset[TRAIN],
            eval_dataset=tokenized_dataset[VALIDATION],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=partial(
                compute_metrics,
                output_type=cfg.LOADER.OUTPUT_TYPE,
                num_classes=cfg.MODEL.NUM_LABELS,
                bin_edges=bin_edges,
            ),
            callbacks=(
                [
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg.TRAIN.PATIENCE,
                        early_stopping_threshold=1e-4,
                    )
                ]
                if cfg.TRAIN.EARLY_STOPPING
                else None
            ),  # NOTE: patience is number of evaluation calls
            optimizers=(optimizer, lr_scheduler),  # TODO
        )
    if cfg.LOADER.OUTPUT_TYPE == OutputType.ORD_LOGIT:
        callback = PrintCutpointsCallback()
        trainer.add_callback(callback)

    ##### TRAINING #####

    logger.info(
        "Train - start",
        epochs=cfg.TRAIN.EPOCHS,
        early_stopping=cfg.TRAIN.EARLY_STOPPING,
    )
    train_start_time = time.time()
    train_metrics = trainer.train()
    train_end_time = time.time()

    logger.info(
        "Train - end",
        train_loss=round(train_metrics.training_loss, ROUND_FLOAT),
        epochs_trained=train_metrics.metrics["epoch"],
        train_time=format_time(elp=train_end_time - train_start_time),
    )
    metrics = {
        **dict(train_metrics.metrics),
        "epochs_trained": train_metrics.metrics[
            "epoch"
        ],  # NOTE: true amount, corrected for max_steps
    }

    ##### EVALUATION #####

    logger.info(
        "Evaluate - start",
        val_size=len(tokenized_dataset[VALIDATION]),
        test_size=len(tokenized_dataset[TEST]),
    )
    eval_start_time = time.time()
    val_results = trainer.predict(
        tokenized_dataset[VALIDATION], metric_key_prefix="val"
    )
    val_metrics = val_results.metrics
    test_results = trainer.predict(tokenized_dataset[TEST], metric_key_prefix="test")
    test_metrics = test_results.metrics
    test_pred_label = (  # NOTE: of test set
        test_results.predictions.squeeze(),
        test_results.label_ids,
    )
    eval_end_time = time.time()
    logger.info(
        "Evaluate - end",
        val_loss=round(val_metrics["val_loss"], ROUND_FLOAT),
        val_bal_rps=round(val_metrics["val_bal_rps"], ROUND_FLOAT),  # NOTE: metric
        test_loss=round(test_metrics["test_loss"], ROUND_FLOAT),
        test_bal_rps=round(test_metrics["test_bal_rps"], ROUND_FLOAT),  # NOTE: metric
        eval_time=format_time(elp=eval_end_time - eval_start_time),
    )
    metrics.update(
        {
            **val_metrics,
            **test_metrics,
        }
    )
    convergence = {
        "log_history": trainer.state.log_history,
    }
    if cfg.LOADER.OUTPUT_TYPE == OutputType.ORD_LOGIT:
        # Access the saved cutpoints after training
        convergence["params_history"] = callback.params_history

    return metrics, convergence, test_pred_label


class PrintCutpointsCallback(TrainerCallback):
    def __init__(self):
        self.params_history = {"cutpoints": [], "betas": []}

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            model = kwargs["model"]
            cutpoints = torch.cumsum(
                torch.cat(
                    (
                        torch.tensor([0.0], device=next(model.parameters()).device),
                        torch.exp(model.classifier.deltas),
                    ),
                    dim=0,
                ),
                dim=0,
            ).tolist()
            self.params_history["cutpoints"].append(cutpoints)
            betas = {
                "bias": model.classifier.linear.bias.tolist(),
                "weights": model.classifier.linear.weight.squeeze().tolist(),
            }
            self.params_history["betas"].append(betas)
            print(f"{cutpoints = }")
            print(f"{betas['bias'] = }")


def get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
    2. By parameter name patterns (containing 'bias', 'layernorm', or 'rmsnorm')
    """
    decay_parameters = get_parameter_names(
        model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"]
    )
    return decay_parameters


def create_optimizer_orderedlogit(
    model: nn.Module,
    args: TrainingArguments,
    lr_multiplier: float,
    params_high_lr_names: list[str],
) -> AdamW:
    """Create an AdamW optimizer for the OrderedLogit model.

    The deltas (cutpoints) are given a higher learning rate than the rest of the model.

    Parameters
    ----------
    model : nn.Module
        Huggingface model with OrderedLogitLayer
    args : TrainingArguments
        Training arguments containing hyperparameters for the optimizer.

    Returns
    -------
    AdamW
        AdamW optimizer.
    """
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    n in decay_parameters
                    and p.requires_grad
                    and n not in params_high_lr_names
                )
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    n not in decay_parameters
                    and p.requires_grad
                    and n not in params_high_lr_names
                )
            ],
            "weight_decay": 0.0,
        },
        {  # NOTE: high lr params do not have weight decay
            "params": [
                p
                for n, p in model.named_parameters()
                if (p.requires_grad and n in params_high_lr_names)
            ],
            "weight_decay": 0.0,
            "lr": args.learning_rate * lr_multiplier,
        },
    ]

    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }

    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


def create_scheduler(
    num_training_steps: int, optimizer: torch.optim.Optimizer, args: TrainingArguments
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=args.lr_scheduler_kwargs,
    )
    return lr_scheduler
