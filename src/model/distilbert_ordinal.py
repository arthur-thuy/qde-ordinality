"""Module for building ordinal DistilBERT."""

# standard library imports
from typing import Optional, Tuple, Union

# related third party imports
import torch
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertModel,
    DistilBertPreTrainedModel,
)
from yacs.config import CfgNode

# local application/library specific imports
from model.build import ORDINAL_MODEL_REGISTRY
from tools.constants import OrdinalType, OutputType

# NOTE: model from `modeling_distilbert.py`
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py


@ORDINAL_MODEL_REGISTRY.register("distilbert-base-uncased")
def build_distilbert_ordinal(
    model_cfg: CfgNode, output_type: OutputType
) -> PreTrainedModel:
    """Build DistilBERT model for ordinal regression.

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
    model = DistilBertForSequenceOrdinal.from_pretrained(
        pretrained_model_name_or_path=model_cfg.NAME,
        num_labels=model_cfg.NUM_LABELS,
        problem_type=output_type,
    )
    return model


class DistilBertForSequenceOrdinal(DistilBertPreTrainedModel):
    """DistilBERT model for ordinal regression."""

    def __init__(self, config: PretrainedConfig):
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
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # output layer
        if config.problem_type == OrdinalType.CORAL:
            # NOTE: layer does num_classes-1 nodes
            self.classifier = CoralLayer(config.dim, config.num_labels)
        elif config.problem_type == OrdinalType.CORN:
            self.classifier = nn.Linear(config.dim, config.num_labels - 1)
        elif config.problem_type == OrdinalType.OR_NN:
            self.classifier = nn.Linear(config.dim, config.num_labels - 1)
        else:
            raise NotImplementedError(f"Output type {config.problem_type} is not implemented")
        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """Return the position embeddings."""
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
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

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type == OutputType.ORD_CORAL:
                levels = levels_from_labelbatch(labels, num_classes=self.num_labels)
                levels = levels.to(logits.device)
                loss = coral_loss(logits, levels)  # NOTE: no need to flatten
            elif self.config.problem_type == OutputType.ORD_CORN:
                loss = corn_loss(
                    logits, labels, num_classes=self.num_labels
                )  # NOTE: no need to flatten
            elif self.config.problem_type == OutputType.ORD_OR_NN:
                levels = levels_from_labelbatch(labels, num_classes=self.num_labels)
                levels = levels.to(logits.device)
                loss = coral_loss(logits, levels)  # NOTE: same loss as CORAL!

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
