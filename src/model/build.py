"""Build file for models."""

# standard library imports
# /

# related third party imports
import structlog
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from yacs.config import CfgNode

# local application/library specific imports
from tools.constants import OrdinalType, OutputType
from tools.registry import Registry

ORDINAL_MODEL_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_model(model_cfg: CfgNode, output_type: OutputType) -> tuple:
    """Build the HuggingFace model and tokenizer.

    Parameters
    ----------
    model_cfg : CfgNode
        Model config object
    output_type : OutputType
        Output type

    Returns
    -------
    tuple
        Tuple of model and tokenizer
    """
    try:
        output_type = OutputType(output_type)
    except ValueError:
        raise ValueError(f"Invalid output type: {output_type}")

    logger.info(
        "Creating model and tokenizer",
        name=model_cfg.NAME,
        output_type=output_type.value,
    )
    if output_type in OrdinalType._value2member_map_:
        model = ORDINAL_MODEL_REGISTRY[model_cfg.NAME](model_cfg, output_type)
    elif output_type == OutputType.REGR:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_cfg.NAME,
            num_labels=1,
            # NOTE: do not use "torch_dtype=torch.bfloat16" for flash attention 2 because causes bad convergence!
            torch_dtype="auto",
        )
    else:
        # NOTE: classification
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_cfg.NAME,
            num_labels=model_cfg.NUM_LABELS,
            # NOTE: do not use "torch_dtype=torch.bfloat16" for flash attention 2 because causes bad convergence!
            torch_dtype="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_cfg.NAME,
        clean_up_tokenization_spaces=True,
    )
    return model, tokenizer
