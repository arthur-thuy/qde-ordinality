"""Module for building ordinal ModernBERT."""

# standard library imports
from typing import Optional, Tuple, Union

# related third party imports
import torch
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertModel,
    ModernBertPreTrainedModel,
    ModernBertPredictionHead,
)
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from yacs.config import CfgNode

# local application/library specific imports
from model.build import ORDINAL_MODEL_REGISTRY
from tools.constants import OrdinalType, OutputType

# NOTE: model from `modeling_modernbert.py`
# https://github.com/huggingface/transformers/blob/fa56dcc2ab748a2d98218b4918742e25454ef0d2/src/transformers/models/modernbert/modeling_modernbert.py#L1192


@ORDINAL_MODEL_REGISTRY.register("answerdotai/ModernBERT-base")
def build_modernbert_ordinal(
    model_cfg: CfgNode, output_type: OutputType
) -> PreTrainedModel:
    """Build ModernBERT model for ordinal regression.

    Parameters
    ----------
    model_cfg : CfgNode
        Model config object
    output_type : OutputType
        Output type

    Returns
    -------
    PreTrainedModel
        HF model
    """
    model = ModernBertForSequenceOrdinal.from_pretrained(
        pretrained_model_name_or_path=model_cfg.NAME,
        num_labels=model_cfg.NUM_LABELS,
        problem_type=output_type,
        torch_dtype="auto",
    )
    return model


class ModernBertForSequenceOrdinal(ModernBertPreTrainedModel):
    """ModernBERT model for ordinal regression."""

    def __init__(self, config: ModernBertConfig):
        """Initialize the model.

        Parameters
        ----------
        config : PretrainedConfig
            _description_

        Raises
        ------
        ValueError
            If `output_type` is not specified in the config
        NotImplementedError
            If the output type is not implemented
        """
        super().__init__(config)
        if self.config.problem_type is None:
            raise ValueError("`output_type` must be specified in the config")
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)

        # output layer
        if config.problem_type == OrdinalType.CORAL:
            # NOTE: layer does num_classes-1 nodes
            self.classifier = CoralLayer(config.hidden_size, config.num_labels)
        elif config.problem_type == OrdinalType.CORN:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels - 1)
        elif config.problem_type == OrdinalType.OR_NN:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels - 1)
        else:
            raise NotImplementedError(
                f"Output type {config.problem_type} is not implemented"
            )

        # Initialize weights and apply final processing
        self.post_init()

    # def _init_weights(self, module):  # TODO for ordinal logit
    #     """Initialize the weights"""
    #     if isinstance(module, OrderedLogitLayer):
    #         # Let OrderedLogitLayer initialize itself
    #         module.reset_parameters()
    #     else:
    #         # Use default BERT initialization for other layers
    #         super()._init_weights(module)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        try:
            self.config.problem_type = OutputType(self.config.problem_type)
        except ValueError:
            raise ValueError(f"Invalid output type: {self.config.problem_type}")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        self._maybe_set_compile()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(
                dim=1
            ) / attention_mask.sum(dim=1, keepdim=True)

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        # print(f"{logits.shape = }")  # TODO: remove
        # print(f"{labels.shape = }")  # TODO: remove

        loss = None
        if labels is not None:
            if self.config.problem_type == OutputType.ORD_CORAL:
                levels = levels_from_labelbatch(labels, num_classes=self.num_labels)
                levels = levels.to(logits.device)
                loss = coral_loss(logits, levels)  # NOTE: no need to flatten
                # print(f"{levels.shape = }")  # TODO: remove
                # print(f"{loss = }")  # TODO: remove
            elif self.config.problem_type == OutputType.ORD_CORN:
                loss = corn_loss(
                    logits, labels, num_classes=self.num_labels
                )  # NOTE: no need to flatten
                # print(f"{loss = }")  # TODO: remove
            elif self.config.problem_type == OutputType.ORD_OR_NN:
                levels = levels_from_labelbatch(labels, num_classes=self.num_labels)
                levels = levels.to(logits.device)
                loss = coral_loss(logits, levels)  # NOTE: same loss as CORAL!
                # print(f"{levels.shape = }")  # TODO: remove
                # print(f"{loss = }")  # TODO: remove

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
