"""Module for building ordinal RoBERTa."""

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
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from yacs.config import CfgNode

# local application/library specific imports
from model.build import ORDINAL_MODEL_REGISTRY
from tools.constants import OrdinalType, OutputType

# NOTE: model from `modeling_bert.py`
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py


@ORDINAL_MODEL_REGISTRY.register("roberta-base")
def build_roberta_ordinal(model_cfg: CfgNode, output_type: OutputType) -> PreTrainedModel:
    """Build RoBERTa model for ordinal regression.

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
    model = RobertaForSequenceOrdinal.from_pretrained(
        pretrained_model_name_or_path=model_cfg.NAME,
        num_labels=model_cfg.NUM_LABELS,
        problem_type=output_type,
    )
    return model


class RobertaOrdinalHead(nn.Module):
    """Head for sentence-level ordinal regression tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # output layer
        if config.problem_type == OrdinalType.CORAL:
            # NOTE: layer does num_classes-1 nodes
            self.out_proj = CoralLayer(config.hidden_size, config.num_labels)
        elif config.problem_type == OrdinalType.CORN:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels - 1)
        elif config.problem_type == OrdinalType.OR_NN:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels - 1)
        else:
            raise NotImplementedError(
                f"Output type {config.problem_type} is not implemented"
            )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceOrdinal(RobertaPreTrainedModel):
    """RoBERTa model for ordinal regression."""

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        if self.config.problem_type is None:
            raise ValueError("`output_type` must be specified in the config")
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaOrdinalHead(config)  # TODO: need to adapt this!

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
